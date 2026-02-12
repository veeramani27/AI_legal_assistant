import { useState, useRef, useEffect, type ChangeEvent } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import { v4 as uuidv4 } from 'uuid';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

function LegalAIChat() {
    const { id: urlId } = useParams<{ id: string }>();
    const navigate = useNavigate();

    // Session/thread id provided by backend; initially empty until first response
    const [sessionThreadId, setSessionThreadId] = useState<string | null>(null);

    const [messages, setMessages] = useState<Message[]>([]);
    const [file, setFile] = useState<File | null>(null);
    const [query, setQuery] = useState<string>("");
    const [loading, setLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);

    const mediaRecorder = useRef<MediaRecorder | null>(null);
    const audioChunks = useRef<Blob[]>([]);
    const chatEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // If URL contains an id param, prefer that as the session id
    useEffect(() => {
        if (urlId) setSessionThreadId(urlId);
    }, [urlId]);

    // Auto-scroll logic
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);


    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setFile(e.target.files[0]);
            toast.success(`Attached: ${e.target.files[0].name}`);
        }
    };

    const triggerFileSelect = () => fileInputRef.current?.click();

    const handleSubmit = async () => {
        if (!query.trim()) return;

        const userMsg: Message = { id: uuidv4(), role: 'user', content: query, timestamp: new Date() };
        setMessages((prev) => [...prev, userMsg]);

        // Capture query to clear state immediately for better UX
        const activeQuery = query;
        setQuery("");
        setLoading(true);

        const formData = new FormData();
        if (file) formData.append("file", file);
        formData.append("user_query", activeQuery);

        // Only include thread_id if we already have one from backend/URL
        if (sessionThreadId) formData.append("thread_id", sessionThreadId);

        try {
            const res = await axios.post("http://localhost:8000/api/text", formData);
            setMessages((prev) => [
                ...prev,
                { id: Math.random().toString(36).slice(2), role: 'assistant', content: res.data.result, timestamp: new Date() },
            ]);

            // Capture backend-generated thread id and update URL if needed
            if (res.data?.thread_id) {
                setSessionThreadId(res.data.thread_id);
                navigate(`/${res.data.thread_id}`, { replace: true });
            }
        } catch {
            toast.error("Failed to get response");
        } finally {
            setLoading(false);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder.current = new MediaRecorder(stream);
            audioChunks.current = [];
            mediaRecorder.current.ondataavailable = (e) => audioChunks.current.push(e.data);
            mediaRecorder.current.onstop = handleVoiceStop;
            mediaRecorder.current.start();
            setIsRecording(true);
        } catch {
            toast.error("Microphone access denied");
        }
    };

    const handleVoiceStop = async () => {
        setLoading(true);
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("audio_file", audioBlob);
        if (file) formData.append("file", file);

        if (sessionThreadId) formData.append("thread_id", sessionThreadId);

        try {
            const res = await axios.post("http://localhost:8000/api/voice", formData);
            const { audio_base64, ai_text, thread_id } = res.data;

            // Handle Audio Playback
            const binary = atob(audio_base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/mpeg" }));
            const audio = new Audio(audioUrl);
            audio.playbackRate = 1.25;
            audio.play();

            setMessages((prev) => [
                ...prev,
                { id: Math.random().toString(36).slice(2), role: 'user', content: "🎤 Voice input", timestamp: new Date() },
                { id: Math.random().toString(36).slice(2), role: 'assistant', content: ai_text, timestamp: new Date() },
            ]);

            // Update session id from backend if provided
            if (thread_id && !sessionThreadId) {
                setSessionThreadId(thread_id);
                navigate(`/${thread_id}`, { replace: true });
            }
        } catch {
            toast.error("Voice processing failed");
        } finally {
            setLoading(false);
        }
    };

    // New Chat handler: simply reloads page to get a fresh constant ID
    const handleNewConsultation = () => {
        window.location.href = '/';
    };

    return (
        <div style={darkThemeStyles.app}>
            <Toaster position="top-center" />

            <aside style={darkThemeStyles.sidebar}>
                <div style={darkThemeStyles.sidebarHeader}>
                    <div style={darkThemeStyles.logo}>
                        <span style={{ fontSize: '1.8rem' }}>⚖️</span>
                        <h1 style={darkThemeStyles.logoText}>LegalAI</h1>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '4px' }}>
                        SESSION ID: {sessionThreadId?.slice(0, 8).toUpperCase()}
                    </div>
                </div>

                <div style={darkThemeStyles.divider} />

                <div style={darkThemeStyles.section}>
                    <div style={darkThemeStyles.sectionHeader}>
                        <span>Context Document</span>
                        {file && (
                            <button onClick={() => setFile(null)} style={darkThemeStyles.removeBtn}>
                                Remove
                            </button>
                        )}
                    </div>

                    <input ref={fileInputRef} type="file" hidden onChange={handleFileChange} accept=".pdf,.txt,.doc,.docx" />

                    {file ? (
                        <div style={darkThemeStyles.fileCard}>
                            <div style={darkThemeStyles.fileRow}>
                                <div style={darkThemeStyles.fileIcon}>📄</div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={darkThemeStyles.fileName}>{file.name}</div>
                                    <div style={darkThemeStyles.fileSize}>{(file.size / 1024).toFixed(1)} KB</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <button onClick={triggerFileSelect} style={darkThemeStyles.dropZone}>
                            <span style={{ fontSize: '2.5rem', opacity: 0.4 }}>☁️</span>
                            <div style={{ marginTop: 12, fontWeight: 500 }}>Upload legal doc</div>
                        </button>
                    )}
                </div>

                <div style={{ flex: 1 }} />

                <button onClick={handleNewConsultation} style={darkThemeStyles.newChatButton}>
                    + New Consultation
                </button>
            </aside>

            <main style={darkThemeStyles.main}>
                <div style={darkThemeStyles.messagesContainer}>
                    {messages.length === 0 ? (
                        <div style={darkThemeStyles.emptyState}>
                            <h2 style={darkThemeStyles.emptyTitle}>Secure Legal Intelligence</h2>
                            <p style={darkThemeStyles.emptySubtitle}>Your session is encrypted and isolated. How can I assist you?</p>
                        </div>
                    ) : (
                        messages.map((msg) => (
                            <div key={msg.id} style={msg.role === 'user' ? darkThemeStyles.userMessageRow : darkThemeStyles.aiMessageRow}>
                                <div style={msg.role === 'user' ? darkThemeStyles.userBubble : darkThemeStyles.aiBubble}>
                                    <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{msg.content}</ReactMarkdown>
                                    <div style={darkThemeStyles.timestamp}>
                                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                    <div ref={chatEndRef} />
                </div>

                <footer style={darkThemeStyles.inputArea}>
                    <div style={darkThemeStyles.inputWrapper}>
                        {/* <button onClick={triggerFileSelect} style={darkThemeStyles.iconButton}>📎</button> */}
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                            placeholder="Ask a legal question..."
                            style={darkThemeStyles.textInput}
                            disabled={loading}
                        />
                        <button
                            onMouseDown={startRecording}
                            onMouseUp={() => { mediaRecorder.current?.stop(); setIsRecording(false); }}
                            style={{ ...darkThemeStyles.iconButton, color: isRecording ? '#f87171' : '#94a3b8' }}
                        >
                            {isRecording ? '🔴' : '🎤'}
                        </button>
                        <button onClick={handleSubmit} disabled={loading || !query.trim()} style={darkThemeStyles.sendButton}>
                            {loading ? '…' : 'Send'}
                        </button>
                    </div>
                </footer>
            </main>
        </div>
    );
}

const darkThemeStyles: Record<string, React.CSSProperties> = {
    app: {
        display: 'flex',
        height: '100vh',
        width: '100vw', // FIX 1: Forces the app to take full viewport width
        background: '#0f172a',
        color: '#e2e8f0',
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        boxSizing: 'border-box',
        overflow: 'hidden',
    },

    // ─── Sidebar ───
    sidebar: {
        width: 280,
        background: '#1e293b',
        borderRight: '1px solid #334155',
        display: 'flex',
        flexDirection: 'column',
        padding: '1.5rem',
        gap: '1rem',
        flexShrink: 0, // Prevents sidebar from shrinking
    },
    sidebarHeader: { display: 'flex', flexDirection: 'column', gap: '1.25rem' },
    logo: { display: 'flex', alignItems: 'center', gap: '0.75rem' },
    logoText: { fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.025em', color: '#60a5fa' },
    threadInfo: { display: 'flex', flexDirection: 'column', gap: '0.25rem' },
    threadLabel: { fontSize: '0.6875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#94a3b8' },
    threadId: { fontFamily: 'ui-monospace, monospace', fontSize: '0.875rem', color: '#cbd5e1', background: '#334155', padding: '0.35rem 0.6rem', borderRadius: '6px', width: 'fit-content' },
    divider: { height: 1, background: '#334155', margin: '0.75rem 0' },
    section: { display: 'flex', flexDirection: 'column', gap: '0.75rem' },
    sectionHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.875rem', fontWeight: 600, color: '#cbd5e1' },
    uploadBtn: { background: '#2563eb', color: 'white', border: 'none', padding: '0.35rem 0.9rem', borderRadius: '6px', fontSize: '0.8125rem', fontWeight: 600, cursor: 'pointer' },
    removeBtn: { background: '#4b1d1d', color: '#fca5a5', border: 'none', padding: '0.35rem 0.8rem', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 500, cursor: 'pointer' },
    fileCard: { background: '#334155', borderRadius: '10px', padding: '1rem', border: '1px solid #475569' },
    fileRow: { display: 'flex', alignItems: 'center', gap: '0.75rem' },
    fileIcon: { fontSize: '1.5rem', opacity: 0.9 },
    fileName: { fontSize: '0.9375rem', fontWeight: 500, color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
    fileSize: { fontSize: '0.75rem', color: '#94a3b8', marginTop: 2 },
    dropZone: { background: '#1e293b', border: '2px dashed #475569', borderRadius: '12px', padding: '2.5rem 1rem', textAlign: 'center' as const, color: '#cbd5e1', fontSize: '0.95rem', cursor: 'pointer', transition: 'all 0.2s' },
    newChatButton: { background: '#2563eb', color: 'white', border: 'none', padding: '0.9rem', borderRadius: '10px', fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer', marginTop: '1rem' },

    main: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        background: '#0f172a',
        width: '100%',
        minWidth: 0,
    },
    messagesContainer: {
        flex: 1,
        width: '100%',
        padding: '2rem 3rem',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.75rem',
        boxSizing: 'border-box',
    },
    emptyState: {
        flex: 1,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center' as const,
    },
    emptyTitle: { fontSize: '2.5rem', fontWeight: 700, color: '#60a5fa', marginBottom: '1rem' },
    emptySubtitle: { fontSize: '1.125rem', color: '#94a3b8', maxWidth: 480 },

    // Messages
    userMessageRow: { display: 'flex', justifyContent: 'flex-end' },
    aiMessageRow: { display: 'flex', justifyContent: 'flex-start' },
    userBubble: { maxWidth: '70%', padding: '1rem 1.25rem', background: '#2563eb', color: 'white', borderRadius: '20px 20px 4px 20px', lineHeight: 1.6, boxShadow: '0 4px 14px rgba(37,99,235,0.25)' },
    aiBubble: { maxWidth: '78%', padding: '1.125rem 1.5rem', background: '#1e293b', color: '#e2e8f0', borderRadius: '20px 20px 20px 4px', lineHeight: 1.65, border: '1px solid #334155', boxShadow: '0 2px 10px rgba(0,0,0,0.2)' },
    timestamp: { fontSize: '0.75rem', opacity: 0.65, marginTop: '0.5rem', textAlign: 'right' as const },

    // Input
    inputArea: {
        padding: '1.25rem 3rem',
        background: '#0f172a',
        borderTop: '1px solid #334155',
        width: '100%', // Ensures footer is full width
        boxSizing: 'border-box',
    },
    inputWrapper: { display: 'flex', alignItems: 'center', gap: '0.75rem', background: '#1e293b', borderRadius: '999px', padding: '0.75rem 1.25rem', border: '1px solid #475569' },
    textInput: { flex: 1, background: 'transparent', border: 'none', outline: 'none', color: '#e2e8f0', fontSize: '1.05rem', padding: '0.5rem 0' },
    iconButton: { background: 'transparent', border: 'none', fontSize: '1.4rem', color: '#94a3b8', cursor: 'pointer', padding: '0.5rem', borderRadius: '50%', transition: 'background 0.15s' },
    sendButton: { background: '#3b82f6', color: 'white', border: 'none', padding: '0.65rem 1.5rem', borderRadius: '999px', fontWeight: 600, cursor: 'pointer', transition: 'background 0.2s' },
    disclaimer: { textAlign: 'center' as const, fontSize: '0.75rem', color: '#64748b', marginTop: '1rem' },
};

export default LegalAIChat;