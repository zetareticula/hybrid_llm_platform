import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [metrics, setMetrics] = useState([]);
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:9000/ws');
    ws.onmessage = e => setMetrics(prev => [...prev.slice(-99), JSON.parse(e.data)]);
    return () => ws.close();
  }, []);

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={metrics}>
        <XAxis dataKey="model" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="latency_ms" stroke="#82ca9d" />
        <Line type="monotone" dataKey="gpu_utilization_pct" stroke="#8884d8" />
      </LineChart>
    </ResponsiveContainer>
  );
}