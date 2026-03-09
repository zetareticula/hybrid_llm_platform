import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, AreaChart, Area,
} from 'recharts';

// ── Constants ────────────────────────────────────────────────────────────────

const WS_URL   = 'ws://localhost:8081/metrics';
const API_URL  = 'http://localhost:8080';
const MAX_HISTORY = 60; // keep last 60 data points on charts

const MODELS = [
  'Llama-3-8B',
  'Llama-3-70B',
  'Mistral-7B',
  'Mixtral-8x7B',
  'Phi-3-Mini',
];

const QUANT_METHODS = [
  { value: 'quantum_nf4', label: 'QuantumNF4 (8×)', color: '#a78bfa' },
  { value: 'nf4',         label: 'NF4 (4×)',        color: '#34d399' },
  { value: 'gptq',        label: 'GPTQ (4×)',       color: '#60a5fa' },
  { value: 'awq',         label: 'AWQ (4×)',        color: '#fbbf24' },
  { value: 'int8',        label: 'INT8 (2×)',       color: '#f87171' },
];

const BACKEND_COLORS = {
  GPU:     '#60a5fa',
  Quantum: '#a78bfa',
  Hybrid:  '#34d399',
  CPU:     '#fbbf24',
};

// ── Helper components ────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: '#1e1e2e', borderRadius: 12, padding: '16px 20px',
      borderLeft: `4px solid ${accent}`, minWidth: 140,
    }}>
      <div style={{ color: '#6b7280', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 28, fontWeight: 700, margin: '4px 0 2px' }}>{value}</div>
      {sub && <div style={{ color: '#6b7280', fontSize: 11 }}>{sub}</div>}
    </div>
  );
}

function Badge({ children, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 600,
    }}>
      {children}
    </span>
  );
}

// ── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [history,      setHistory]      = useState([]);
  const [tokenStream,  setTokenStream]  = useState([]);
  const [wsStatus,     setWsStatus]     = useState('connecting');
  const [totalTokens,  setTotalTokens]  = useState(0);
  const [avgTps,       setAvgTps]       = useState(0);
  const [backendDist,  setBackendDist]  = useState([]);
  const [prompt,       setPrompt]       = useState('');
  const [model,        setModel]        = useState(MODELS[0]);
  const [quant,        setQuant]        = useState('quantum_nf4');
  const [sending,      setSending]      = useState(false);
  const [lastResponse, setLastResponse] = useState(null);

  const streamRef  = useRef(null);
  const wsRef      = useRef(null);

  // ── WebSocket — live metrics from port 8081 ───────────────────────────────
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen  = () => setWsStatus('live');
      ws.onclose = () => { setWsStatus('reconnecting'); setTimeout(connect, 2000); };
      ws.onerror = () => setWsStatus('error');

      ws.onmessage = e => {
        try {
          const m = JSON.parse(e.data);
          const ts = Date.now();

          setHistory(prev => {
            const next = [...prev.slice(-(MAX_HISTORY - 1)), {
              t:          ts,
              label:      new Date(ts).toLocaleTimeString(),
              latency_ms: m.latency_ms ?? 0,
              tps:        m.tps        ?? 0,
              tokens:     m.tokens     ?? 0,
            }];
            return next;
          });

          setTotalTokens(n => n + (m.tokens ?? 0));

          setAvgTps(prev => {
            const tps = m.tps ?? 0;
            return prev === 0 ? tps : prev * 0.8 + tps * 0.2; // EMA
          });

          if (m.backend) {
            setBackendDist(prev => {
              const existing = prev.find(b => b.name === m.backend);
              if (existing) {
                return prev.map(b => b.name === m.backend ? { ...b, value: b.value + 1 } : b);
              }
              return [...prev, { name: m.backend, value: 1 }];
            });
          }

          if (m.tokens) {
            const tags = Array.from({ length: m.tokens }, (_, i) => `tok${i + 1}`);
            setTokenStream(prev => [...prev.slice(-199), ...tags]);
          }
        } catch (_) {}
      };
    }
    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── Send prompt to REST API ────────────────────────────────────────────────
  const sendPrompt = useCallback(async () => {
    if (!prompt.trim() || sending) return;
    setSending(true);
    setLastResponse(null);
    try {
      const res = await fetch(`${API_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          quantization:     quant,
          quantum_provider: quant === 'quantum_nf4' ? 'ibm' : undefined,
          backend:          quant === 'quantum_nf4' ? 'quantum' : 'hybrid',
          messages: [{ role: 'user', content: prompt }],
        }),
      });
      const data = await res.json();
      setLastResponse(data);
      setTokenStream(prev => [...prev.slice(-199), ...(data.token_stream ?? [])]);
    } catch (err) {
      setLastResponse({ error: err.message });
    } finally {
      setSending(false);
    }
  }, [prompt, model, quant, sending]);

  // ── Derived values ─────────────────────────────────────────────────────────
  const lastLatency  = history.at(-1)?.latency_ms ?? 0;
  const activeQuant  = QUANT_METHODS.find(q => q.value === quant);
  const comprRatio   = { quantum_nf4: '8×', nf4: '4×', gptq: '4×', awq: '4×', int8: '2×' }[quant] ?? '?';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: '100vh', background: '#0f0f1a', color: '#f1f5f9',
      fontFamily: '"Inter", system-ui, sans-serif', padding: '24px 32px',
    }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>
            ⚛ Hybrid LLM Platform
          </h1>
          <div style={{ color: '#6b7280', fontSize: 13, marginTop: 4 }}>
            Quantum-compressed open-source inference · 10× cheaper than proprietary APIs
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <Badge color={wsStatus === 'live' ? '#34d399' : wsStatus === 'reconnecting' ? '#fbbf24' : '#f87171'}>
            {wsStatus === 'live' ? '● LIVE' : wsStatus === 'reconnecting' ? '◌ RECONNECTING' : '✕ ERROR'}
          </Badge>
          <Badge color='#a78bfa'>{model}</Badge>
          <Badge color={activeQuant?.color ?? '#6b7280'}>{activeQuant?.label}</Badge>
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 28 }}>
        <StatCard label='Total Tokens'    value={totalTokens.toLocaleString()} sub='lifetime'         accent='#60a5fa' />
        <StatCard label='Throughput'      value={`${avgTps.toFixed(0)}`}        sub='tokens / second'  accent='#34d399' />
        <StatCard label='Last Latency'    value={`${lastLatency}ms`}            sub='end-to-end'       accent='#fbbf24' />
        <StatCard label='Compression'     value={comprRatio}                    sub='vs FP16 baseline' accent='#a78bfa' />
        <StatCard label='Active Backends' value={backendDist.length || '—'}     sub='GPU · Quantum · Hybrid' accent='#f87171' />
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20, marginBottom: 28 }}>

        {/* Throughput + latency area chart */}
        <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 20 }}>
          <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 12 }}>THROUGHPUT &amp; LATENCY — last {MAX_HISTORY} events</div>
          <ResponsiveContainer width='100%' height={200}>
            <AreaChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id='tpsGrad' x1='0' y1='0' x2='0' y2='1'>
                  <stop offset='5%'  stopColor='#34d399' stopOpacity={0.3} />
                  <stop offset='95%' stopColor='#34d399' stopOpacity={0} />
                </linearGradient>
                <linearGradient id='latGrad' x1='0' y1='0' x2='0' y2='1'>
                  <stop offset='5%'  stopColor='#fbbf24' stopOpacity={0.3} />
                  <stop offset='95%' stopColor='#fbbf24' stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey='label' tick={{ fill: '#4b5563', fontSize: 10 }} interval='preserveStartEnd' />
              <YAxis tick={{ fill: '#4b5563', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#2d2d3d', border: 'none', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Area type='monotone' dataKey='tps'        stroke='#34d399' fill='url(#tpsGrad)' name='tok/s'  strokeWidth={2} dot={false} />
              <Area type='monotone' dataKey='latency_ms' stroke='#fbbf24' fill='url(#latGrad)' name='ms'     strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Backend distribution pie */}
        <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 20 }}>
          <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 12 }}>BACKEND DISTRIBUTION</div>
          {backendDist.length === 0 ? (
            <div style={{ color: '#4b5563', fontSize: 13, textAlign: 'center', paddingTop: 60 }}>
              Waiting for requests…
            </div>
          ) : (
            <ResponsiveContainer width='100%' height={200}>
              <PieChart>
                <Pie data={backendDist} dataKey='value' nameKey='name'
                  cx='50%' cy='50%' outerRadius={70} paddingAngle={3}>
                  {backendDist.map(entry => (
                    <Cell key={entry.name} fill={BACKEND_COLORS[entry.name] ?? '#6b7280'} />
                  ))}
                </Pie>
                <Legend iconType='circle' wrapperStyle={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#2d2d3d', border: 'none', borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Prompt input */}
      <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 12 }}>SEND INFERENCE REQUEST</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>

          <select value={model} onChange={e => setModel(e.target.value)}
            style={inputStyle}>
            {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
          </select>

          <select value={quant} onChange={e => setQuant(e.target.value)}
            style={inputStyle}>
            {QUANT_METHODS.map(q => <option key={q.value} value={q.value}>{q.label}</option>)}
          </select>
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          <input
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendPrompt()}
            placeholder='Ask anything… (Enter to send)'
            style={{ ...inputStyle, flex: 1, fontSize: 14 }}
          />
          <button onClick={sendPrompt} disabled={sending || !prompt.trim()}
            style={{
              background: sending ? '#374151' : '#6d28d9',
              color: '#f1f5f9', border: 'none', borderRadius: 8,
              padding: '10px 24px', cursor: sending ? 'not-allowed' : 'pointer',
              fontWeight: 600, fontSize: 14, transition: 'background .2s',
            }}>
            {sending ? 'Sending…' : 'Run Inference'}
          </button>
        </div>
      </div>

      {/* Token stream */}
      <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 12 }}>
          LIVE TOKEN STREAM — last {tokenStream.length} tokens
        </div>
        <div ref={streamRef} style={{
          fontFamily: '"JetBrains Mono", monospace', fontSize: 13,
          lineHeight: 1.8, color: '#d1d5db', maxHeight: 140, overflowY: 'auto',
          wordBreak: 'break-all',
        }}>
          {tokenStream.length === 0
            ? <span style={{ color: '#4b5563' }}>No tokens yet — send a prompt or start the server</span>
            : tokenStream.map((tok, i) => (
              <span key={i} style={{
                marginRight: 6,
                color: tok.includes('[Q]') ? '#a78bfa'
                      : tok.includes('[GPU]') ? '#60a5fa'
                      : '#d1d5db',
              }}>
                {tok}
              </span>
            ))
          }
        </div>
      </div>

      {/* Last API response */}
      {lastResponse && (
        <div style={{ background: '#1e1e2e', borderRadius: 12, padding: 20, marginBottom: 20 }}>
          <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 8 }}>LAST API RESPONSE</div>
          <pre style={{
            margin: 0, fontFamily: '"JetBrains Mono", monospace',
            fontSize: 12, color: '#d1d5db', overflowX: 'auto',
          }}>
            {JSON.stringify(lastResponse, null, 2)}
          </pre>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#374151', fontSize: 12, paddingTop: 8 }}>
        Hybrid LLM Platform · ws://localhost:8081/metrics · http://localhost:8080
        &nbsp;·&nbsp;
        <span style={{ color: '#a78bfa' }}>⚛ Quantum-assisted NF4 compression</span>
      </div>
    </div>
  );
}

// ── Shared styles ─────────────────────────────────────────────────────────────

const inputStyle = {
  background: '#2d2d3d', border: '1px solid #374151', borderRadius: 8,
  color: '#f1f5f9', padding: '10px 12px', fontSize: 13,
  outline: 'none', cursor: 'pointer',
};