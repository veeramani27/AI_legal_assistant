import { useState, useRef, useEffect, type ChangeEvent, type FormEvent } from 'react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";

function App01() {
    const [file, setFile] = useState<File | null>(null);
    const [query, setQuery] = useState<string>("Is termination for arbitrary cause allowed?");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);

    // Voice & Memory State
    const [isRecording, setIsRecording] = useState(false);
    const [threadId, setThreadId] = useState<string | null>(null);
    const mediaRecorder = useRef<MediaRecorder | null>(null);
    const audioChunks = useRef<Blob[]>([]);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) setFile(e.target.files[0]);
    };

    // --- TEXT SUBMISSION ---
    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setLoading(true);
        const formData = new FormData();
        if (file) formData.append("file", file);
        formData.append("user_query", query);
        if (threadId) formData.append("thread_id", threadId); // Pass thread_id for memory when available

        try {
            const res = await axios.post("http://localhost:8000/api/text", formData);
            setResult(res.data);
            // capture backend generated thread id
            if (res.data?.thread_id) setThreadId(res.data.thread_id);
            toast.success("Analysis complete!");
        } catch (err: any) {
            toast.error("Text analysis failed");
        } finally {
            setLoading(false);
        }
    };

    // --- VOICE SUBMISSION ---
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder.current = new MediaRecorder(stream);
            audioChunks.current = [];

            mediaRecorder.current.ondataavailable = (e) => audioChunks.current.push(e.data);
            mediaRecorder.current.onstop = handleVoiceStop;

            mediaRecorder.current.start();
            setIsRecording(true);
        } catch (err) {
            toast.error("Microphone access denied");
        }
    };

    const stopRecording = () => {
        mediaRecorder.current?.stop();
        setIsRecording(false);
    };

    const handleVoiceStop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        setLoading(true);

        const formData = new FormData();
        formData.append("audio_file", audioBlob);
        if (threadId) formData.append("thread_id", threadId);
        if (file) formData.append("file", file);

        try {
            // Use responseType: 'blob' to handle the audio file coming back
            const res = await axios.post("http://localhost:8000/api/voice", formData, {
                responseType: 'blob',
            });

            // capture thread id from response headers if backend included it
            const headerThread = (res.headers && (res.headers['x-thread-id'] || res.headers['X-Thread-ID'])) || null;
            if (headerThread) setThreadId(headerThread as string);

            // 1. Play the audio response
            const audioUrl = URL.createObjectURL(res.data);
            const audio = new Audio(audioUrl);
            audio.play();

            // 2. Extract AI Text from custom header (see backend logic)
            const aiText = res.headers["x-ai-text"];
            setResult({ result: aiText || "Voice response received." });

            toast.success("Voice response played!");
        } catch (err) {
            toast.error("Voice processing failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: "800px", margin: "2rem auto", padding: "1rem", fontFamily: "sans-serif" }}>
            <Toaster position="top-center" />
            <h1>Legal Contract Analyzer</h1>
            <p><small>Session ID: {threadId}</small></p>

            <div style={{ border: "1px solid #eee", padding: "1rem", borderRadius: "8px" }}>
                <label>Step 1: Upload contract (Optional):</label><br />
                <input type="file" accept=".pdf,.docx" onChange={handleFileChange} />
            </div>

            <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
                <label>Step 2: Ask via Text:</label><br />
                <div style={{ display: "flex", gap: "10px" }}>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        style={{ flex: 1, padding: "0.5rem" }}
                    />
                    <button type="submit" disabled={loading}>
                        {loading ? "..." : "Send Text"}
                    </button>
                </div>
            </form>

            <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
                <p>OR Ask via Voice:</p>
                <button
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onTouchStart={startRecording} // For mobile support
                    onTouchEnd={stopRecording}
                    style={{
                        padding: "1rem 2rem",
                        fontSize: "1.2rem",
                        backgroundColor: isRecording ? "#ff4d4d" : "#007bff",
                        color: "white",
                        border: "none",
                        borderRadius: "50px",
                        cursor: "pointer",
                        width: "100%"
                    }}
                >
                    {isRecording ? "Listening... (Release to Send)" : "Hold to Talk"}
                </button>
            </div>

            {result && (
                <div style={{ marginTop: "2rem", border: "1px solid #ccc", padding: "1rem", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                    <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
                        {result.result}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    );
}

export default App01;