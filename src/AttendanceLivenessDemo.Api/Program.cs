using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);
builder.Services.Configure<AiWorkerOptions>(builder.Configuration.GetSection("AiWorker"));
builder.Services.Configure<LivenessOptions>(builder.Configuration.GetSection("Liveness"));
builder.Services.AddHttpClient<AiWorkerClient>();
builder.Services.AddSingleton<SessionStore>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();
app.UseWebSockets(new WebSocketOptions
{
    KeepAliveInterval = TimeSpan.FromSeconds(15)
});

app.MapGet("/health", () => Results.Ok(new { ok = true, service = "AttendanceLivenessDemo.Api" }));

app.MapPost("/api/session/start", (SessionStore store, HttpContext context) =>
{
    var session = store.Create();
    var scheme = context.Request.IsHttps ? "wss" : "ws";
    var wsUrl = $"{scheme}://{context.Request.Host}/ws/attendance-liveness?sessionId={session.SessionId}";

    return Results.Ok(new
    {
        session.SessionId,
        ExpiresAt = session.ExpiresAt,
        Challenge = session.Steps.Select(x => new
        {
            Type = x.Type.ToString(),
            Title = x.Title,
            Instruction = x.Instruction
        }),
        WsUrl = wsUrl
    });
});

app.Map("/ws/attendance-liveness", async (HttpContext context, SessionStore store, AiWorkerClient aiClient, IOptions<LivenessOptions> options) =>
{
    if (!context.WebSockets.IsWebSocketRequest)
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
        await context.Response.WriteAsync("Expected WebSocket request.");
        return;
    }

    var sessionId = context.Request.Query["sessionId"].ToString();
    if (string.IsNullOrWhiteSpace(sessionId) || !store.TryGet(sessionId, out var session))
    {
        context.Response.StatusCode = StatusCodes.Status404NotFound;
        await context.Response.WriteAsync("Session not found. Call POST /api/session/start first.");
        return;
    }

    using var ws = await context.WebSockets.AcceptWebSocketAsync();
    await WsSendJson(ws, ServerMessage.Progress(session, "CONNECTED", "Đã kết nối WebSocket. Đưa mặt vào giữa khung."), context.RequestAborted);

    var receiveBuffer = new byte[1024 * 512];
    var jsonOptions = AppJson.Options;

    while (ws.State == WebSocketState.Open && !context.RequestAborted.IsCancellationRequested)
    {
        using var ms = new MemoryStream();
        WebSocketReceiveResult result;

        do
        {
            result = await ws.ReceiveAsync(new ArraySegment<byte>(receiveBuffer), context.RequestAborted);

            if (result.MessageType == WebSocketMessageType.Close)
            {
                await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closed", context.RequestAborted);
                return;
            }

            ms.Write(receiveBuffer, 0, result.Count);
        }
        while (!result.EndOfMessage);

        var payload = ms.ToArray();

        if (session.IsExpired)
        {
            await WsSendJson(ws, ServerMessage.Final(session, false, "SESSION_EXPIRED", "Phiên xác thực đã hết hạn."), context.RequestAborted);
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "expired", context.RequestAborted);
            return;
        }

        if (result.MessageType == WebSocketMessageType.Text)
        {
            var text = Encoding.UTF8.GetString(payload);
            ClientMessage? msg = null;

            try { msg = JsonSerializer.Deserialize<ClientMessage>(text, jsonOptions); }
            catch { /* ignore malformed JSON */ }

            if (msg?.Type == "START")
            {
                session.ClientStartedAt = DateTimeOffset.UtcNow;
                await WsSendJson(ws, ServerMessage.Progress(session, "RUNNING", session.CurrentStep.Instruction), context.RequestAborted);
            }
            else if (msg?.Type == "FRAME_META")
            {
                session.LastSeq = msg.Seq;
                session.LastClientTimestampMs = msg.TimestampMs;
            }

            continue;
        }

        if (result.MessageType != WebSocketMessageType.Binary)
        {
            continue;
        }

        session.TotalFrames++;

        AiFrameAnalysis analysis;
        try
        {
            analysis = await aiClient.AnalyzeAsync(payload, session.SessionId, session.LastSeq, context.RequestAborted);
        }
        catch (Exception ex)
        {
            await WsSendJson(ws, new
            {
                type = "ERROR",
                code = "AI_WORKER_ERROR",
                message = "Không gọi được AI worker. Kiểm tra service Python ở port 8001.",
                detail = ex.Message
            }, context.RequestAborted);
            continue;
        }

        var decision = session.Update(analysis, options.Value);

        if (decision.IsFinal)
        {
            await WsSendJson(ws, ServerMessage.Final(session, decision.Passed, decision.Code, decision.Message, analysis), context.RequestAborted);
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, decision.Passed ? "passed" : "failed", context.RequestAborted);
            return;
        }

        await WsSendJson(ws, ServerMessage.Progress(session, decision.Code, decision.Message, analysis), context.RequestAborted);
    }
});

app.Run();

static async Task WsSendJson(WebSocket ws, object data, CancellationToken cancellationToken)
{
    var json = JsonSerializer.Serialize(data, AppJson.Options);
    var bytes = Encoding.UTF8.GetBytes(json);
    await ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cancellationToken);
}

public sealed class AiWorkerOptions
{
    public string AnalyzeUrl { get; set; } = "http://127.0.0.1:8001/analyze";
    public int TimeoutMs { get; set; } = 1200;
}

public sealed class LivenessOptions
{
    public int FrameUploadFps { get; set; } = 6;
    public int MaxSessionSeconds { get; set; } = 35;
    public double MinLivenessScore { get; set; } = 0.55;
    public double MaxSpoofScore { get; set; } = 0.55;
    public string PassiveSpoofMode { get; set; } = "Warn";
    public int PassiveSpoofWarmupFrames { get; set; } = 8;
    public int PassiveSpoofFailFrames { get; set; } = 3;
    public int CenterFaceFrames { get; set; } = 8;
    public int TurnFrames { get; set; } = 4;
    public double YawDegrees { get; set; } = 15.0;
    public int BlinkCount { get; set; } = 2;
    public bool StrictPassiveSpoof => string.Equals(PassiveSpoofMode, "Strict", StringComparison.OrdinalIgnoreCase);
}

public sealed class AiWorkerClient
{
    private readonly HttpClient _http;
    private readonly AiWorkerOptions _options;

    public AiWorkerClient(HttpClient http, IOptions<AiWorkerOptions> options)
    {
        _http = http;
        _options = options.Value;
        _http.Timeout = TimeSpan.FromMilliseconds(_options.TimeoutMs);
    }

    public async Task<AiFrameAnalysis> AnalyzeAsync(byte[] jpegBytes, string sessionId, int? seq, CancellationToken cancellationToken)
    {
        using var form = new MultipartFormDataContent();
        var image = new ByteArrayContent(jpegBytes);
        image.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
        form.Add(image, "file", "frame.jpg");
        form.Add(new StringContent(sessionId), "sessionId");
        form.Add(new StringContent(seq?.ToString() ?? ""), "seq");

        using var response = await _http.PostAsync(_options.AnalyzeUrl, form, cancellationToken);
        response.EnsureSuccessStatusCode();

        var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var result = await JsonSerializer.DeserializeAsync<AiFrameAnalysis>(stream, AppJson.Options, cancellationToken);
        return result ?? new AiFrameAnalysis { FaceFound = false, Message = "Empty AI response" };
    }
}

public static class AppJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);
}

public sealed class SessionStore
{
    private readonly ConcurrentDictionary<string, LivenessSession> _sessions = new();

    public LivenessSession Create()
    {
        var steps = ChallengeFactory.CreateRandom();
        var session = new LivenessSession($"att_{DateTimeOffset.UtcNow:yyyyMMddHHmmss}_{Guid.NewGuid():N}"[..36], steps);
        _sessions[session.SessionId] = session;
        return session;
    }

    public bool TryGet(string sessionId, [NotNullWhen(true)] out LivenessSession? session) => _sessions.TryGetValue(sessionId, out session);
}

public static class ChallengeFactory
{
    private static readonly Random Rng = new();

    public static IReadOnlyList<ChallengeStep> CreateRandom()
    {
        var middle = new List<ChallengeStep>
        {
            ChallengeStep.TurnLeft(),
            ChallengeStep.TurnRight(),
            ChallengeStep.BlinkTwice()
        };

        lock (Rng)
        {
            middle = middle.OrderBy(_ => Rng.Next()).ToList();
        }

        return new[] { ChallengeStep.CenterFace() }.Concat(middle).ToArray();
    }
}

public sealed class LivenessSession
{
    public LivenessSession(string sessionId, IReadOnlyList<ChallengeStep> steps)
    {
        SessionId = sessionId;
        Steps = steps;
        ExpiresAt = DateTimeOffset.UtcNow.AddSeconds(35);
    }

    public string SessionId { get; }
    public IReadOnlyList<ChallengeStep> Steps { get; }
    public DateTimeOffset CreatedAt { get; } = DateTimeOffset.UtcNow;
    public DateTimeOffset ExpiresAt { get; }
    public DateTimeOffset? ClientStartedAt { get; set; }
    public int CurrentStepIndex { get; private set; }
    public int CurrentSatisfiedFrames { get; private set; }
    public int TotalFrames { get; set; }
    public int ValidFaceFrames { get; private set; }
    public int? LastSeq { get; set; }
    public long? LastClientTimestampMs { get; set; }
    public int BlinkCount { get; private set; }
    public bool PreviousEyesOpen { get; private set; } = true;
    public double BestLivenessScore { get; private set; }
    public double WorstSpoofScore { get; private set; }
    public int ConsecutiveSpoofRiskFrames { get; private set; }
    public AiFrameAnalysis? LastAnalysis { get; private set; }
    public bool IsExpired => DateTimeOffset.UtcNow > ExpiresAt;
    public bool IsPassed => CurrentStepIndex >= Steps.Count;
    public ChallengeStep CurrentStep => IsPassed ? Steps[^1] : Steps[CurrentStepIndex];

    public Decision Update(AiFrameAnalysis analysis, LivenessOptions options)
    {
        LastAnalysis = analysis;
        BestLivenessScore = Math.Max(BestLivenessScore, analysis.LivenessScore);
        WorstSpoofScore = Math.Max(WorstSpoofScore, analysis.SpoofScore);

        if (analysis.FaceFound && analysis.FaceCount == 1)
        {
            ValidFaceFrames++;
        }

        if (analysis.FaceCount > 1)
        {
            CurrentSatisfiedFrames = 0;
            return Decision.Progress("MULTI_FACE", "Chỉ được có 1 khuôn mặt trong khung hình.");
        }

        if (!analysis.FaceFound)
        {
            CurrentSatisfiedFrames = 0;
            return Decision.Progress("NO_FACE", "Không thấy khuôn mặt. Đưa mặt vào giữa khung.");
        }

        if (analysis.QualityScore < 0.35)
        {
            CurrentSatisfiedFrames = 0;
            return Decision.Progress("LOW_QUALITY", "Ảnh đang mờ hoặc thiếu sáng. Giữ máy ổn định và tăng ánh sáng.");
        }

        var pastSpoofWarmup = TotalFrames > Math.Max(0, options.PassiveSpoofWarmupFrames);
        if (pastSpoofWarmup && analysis.ModelLoaded && analysis.SpoofScore > options.MaxSpoofScore)
        {
            ConsecutiveSpoofRiskFrames++;
        }
        else
        {
            ConsecutiveSpoofRiskFrames = 0;
        }

        // Demo default is Warn: active challenge proves liveness while ONNX remains visible telemetry.
        if (options.StrictPassiveSpoof &&
            ConsecutiveSpoofRiskFrames >= Math.Max(1, options.PassiveSpoofFailFrames))
        {
            return Decision.Final(false, "SPOOF_RISK", "Phát hiện rủi ro ảnh/video/màn hình giả mạo trong nhiều frame liên tiếp. Vui lòng thử lại với camera thật.");
        }

        var step = CurrentStep;
        var stepOk = false;

        switch (step.Type)
        {
            case ChallengeStepType.CenterFace:
                stepOk = Math.Abs(analysis.Yaw) < 12 && Math.Abs(analysis.Pitch) < 18;
                break;

            case ChallengeStepType.TurnLeft:
                stepOk = analysis.Yaw <= -options.YawDegrees;
                break;

            case ChallengeStepType.TurnRight:
                stepOk = analysis.Yaw >= options.YawDegrees;
                break;

            case ChallengeStepType.BlinkTwice:
                var eyesOpen = analysis.EyeAspectRatio >= 0.20;
                var blinkEdge = PreviousEyesOpen && !eyesOpen;
                if (blinkEdge || analysis.Blink)
                {
                    BlinkCount++;
                }
                PreviousEyesOpen = eyesOpen;
                stepOk = BlinkCount >= options.BlinkCount;
                break;
        }

        if (stepOk)
        {
            CurrentSatisfiedFrames++;
        }
        else if (step.Type != ChallengeStepType.BlinkTwice)
        {
            CurrentSatisfiedFrames = 0;
        }

        var requiredFrames = step.Type switch
        {
            ChallengeStepType.CenterFace => options.CenterFaceFrames,
            ChallengeStepType.BlinkTwice => 1,
            _ => options.TurnFrames
        };

        if (CurrentSatisfiedFrames >= requiredFrames)
        {
            CurrentStepIndex++;
            CurrentSatisfiedFrames = 0;

            if (CurrentStepIndex >= Steps.Count)
            {
                if (options.StrictPassiveSpoof && analysis.ModelLoaded && BestLivenessScore < options.MinLivenessScore)
                {
                    return Decision.Final(false, "LOW_LIVENESS", "Liveness score chưa đủ. Hãy thử lại, không dùng ảnh/video/màn hình.");
                }

                return Decision.Final(true, "PASSED", "Xác thực sống thành công. Có thể ghi nhận chấm công.");
            }

            if (step.Type == ChallengeStepType.BlinkTwice)
            {
                BlinkCount = 0;
            }

            return Decision.Progress("NEXT_STEP", CurrentStep.Instruction);
        }

        return Decision.Progress("RUNNING", step.Instruction);
    }
}

public sealed record ChallengeStep(ChallengeStepType Type, string Title, string Instruction)
{
    public static ChallengeStep CenterFace() => new(ChallengeStepType.CenterFace, "Nhìn thẳng", "Đưa mặt vào giữa khung và nhìn thẳng.");
    public static ChallengeStep TurnLeft() => new(ChallengeStepType.TurnLeft, "Quay trái", "Quay mặt sang trái một chút.");
    public static ChallengeStep TurnRight() => new(ChallengeStepType.TurnRight, "Quay phải", "Quay mặt sang phải một chút.");
    public static ChallengeStep BlinkTwice() => new(ChallengeStepType.BlinkTwice, "Chớp mắt", "Chớp mắt 2 lần.");
}

public enum ChallengeStepType
{
    CenterFace,
    TurnLeft,
    TurnRight,
    BlinkTwice
}

public sealed record Decision(bool IsFinal, bool Passed, string Code, string Message)
{
    public static Decision Progress(string code, string message) => new(false, false, code, message);
    public static Decision Final(bool passed, string code, string message) => new(true, passed, code, message);
}

public sealed class ClientMessage
{
    public string? Type { get; set; }
    public int? Seq { get; set; }
    public long? TimestampMs { get; set; }
    public string? EmployeeId { get; set; }
    public string? DeviceId { get; set; }
}

public sealed class AiFrameAnalysis
{
    public bool FaceFound { get; set; }
    public int FaceCount { get; set; }
    public double Yaw { get; set; }
    public double Pitch { get; set; }
    public double Roll { get; set; }
    public bool Blink { get; set; }
    public double EyeAspectRatio { get; set; }
    public double LivenessScore { get; set; }
    public double SpoofScore { get; set; }
    public double QualityScore { get; set; }
    public bool ModelLoaded { get; set; }
    public string? Message { get; set; }
    public Dictionary<string, double>? Metrics { get; set; }
}

public static class ServerMessage
{
    public static object Progress(LivenessSession session, string code, string message, AiFrameAnalysis? analysis = null) => new
    {
        type = "PROGRESS",
        code,
        message,
        sessionId = session.SessionId,
        currentStepIndex = Math.Min(session.CurrentStepIndex, session.Steps.Count - 1),
        totalSteps = session.Steps.Count,
        currentAction = session.IsPassed ? "DONE" : session.CurrentStep.Type.ToString(),
        currentTitle = session.IsPassed ? "Hoàn tất" : session.CurrentStep.Title,
        satisfiedFrames = session.CurrentSatisfiedFrames,
        totalFrames = session.TotalFrames,
        validFaceFrames = session.ValidFaceFrames,
        spoofRiskFrames = session.ConsecutiveSpoofRiskFrames,
        expiresAt = session.ExpiresAt,
        analysis
    };

    public static object Final(LivenessSession session, bool passed, string code, string message, AiFrameAnalysis? analysis = null) => new
    {
        type = "FINAL",
        passed,
        status = passed ? "PASSED" : "FAILED",
        code,
        message,
        sessionId = session.SessionId,
        totalFrames = session.TotalFrames,
        validFaceFrames = session.ValidFaceFrames,
        spoofRiskFrames = session.ConsecutiveSpoofRiskFrames,
        bestLivenessScore = session.BestLivenessScore,
        worstSpoofScore = session.WorstSpoofScore,
        analysis
    };
}
