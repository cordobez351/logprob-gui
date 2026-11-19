import { useEffect, useRef, useState } from 'react';

import Lightfall from './rb/Lightfall/Lightfall.jsx';
import SpecularButton from './rb/SpecularButton/SpecularButton.jsx';
import JellyRadio from './rb/JellyRadio/JellyRadio.jsx';
import GlideSelect from './rb/GlideSelect/GlideSelect.jsx';
import StatusMark from './rb/StatusMark/StatusMark.jsx';
import Counter from './rb/Counter/Counter.jsx';

import Static from './Static.jsx';
import { Projection, Timeline, Worlds } from './Projection.jsx';
import { meanEntropy, meanTop1 } from './logprobs.js';
import { consumeOpenRouterSse } from './stream.js';
import { T } from './i18n.js';

// Devs: a black room, gold light, and a screen where the future clears out of the noise.
const C = { black: '#050403', gold: '#d9a441', pale: '#f3d58a', dim: '#3a2c12', chip: '#15110a', high: '#241b0c', ink: '#efe6d2', red: '#e5533d' };

// The server proxy allows only these (all return logprobs); keep in sync with /opt/ideitas/logprob/app.py.
const MODELS = [
  'deepseek/deepseek-v4-flash',
  'openai/gpt-4o-mini',
  'meta-llama/llama-3.2-3b-instruct',
  'google/gemma-3-27b-it',
  'qwen/qwen3-30b-a3b-instruct-2507',
  'gryphe/mythomax-l2-13b',
  'ibm-granite/granite-4.0-h-micro',
  'thedrummer/cydonia-24b-v4.1',
];
const TEMPS = [0.2, 0.7, 1.2, 2];
// Error codes from the proxy -> message keys.
const ERRORS = { rate: 'errRate', day: 'errDay', long: 'errLong', upstream: 'errUpstream' };
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// Some models (OpenAI) do not continue an assistant message: they write the answer again
// from the start, keeping the forced token. Cut that echo so only the new future is left.
function stripEcho(prefix, p) {
  const pre = prefix.trimStart();
  const echo = p.text.length < pre.length ? pre.startsWith(p.text) : p.text.startsWith(pre);
  if (!echo || !pre) return p;
  const cut = Math.min(p.text.length, pre.length);
  let n = 0, k = 0;
  while (k < p.steps.length && n + p.steps[k].token.length <= cut) { n += p.steps[k].token.length; k++; }
  return { text: p.text.slice(cut), steps: p.steps.slice(k) };
}

// A "line" is one future for the last question: the original answer, or a branch that
// replaced the token at step `at` with `alt` and let the model continue from there.
// { steps, text, question, ctx (messages sent), at, alt }

export default function App() {
  const [lang, setLang] = useState(store.get('logprob.lang') === 'en' ? 'en' : 'es');
  const t = (k, ...a) => { const v = T[lang][k]; return typeof v === 'function' ? v(...a) : v; };

  const [model, setModel] = useState(MODELS[0]);
  const [temp, setTemp] = useState(0.7);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [lines, setLines] = useState([]);
  const [active, setActive] = useState(0);
  const [live, setLive] = useState(null); // { steps, text, question } while streaming
  const [loading, setLoading] = useState(false);
  const [waiting, setWaiting] = useState(false); // sent, no token yet
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(0);
  const [autoWalk, setAutoWalk] = useState(true);
  const abortRef = useRef(null);
  const partialRef = useRef({ text: '', steps: [] });
  const screenRef = useRef(null);
  const replyRef = useRef(null);

  useEffect(() => { document.documentElement.lang = lang; store.set('logprob.lang', lang); }, [lang]);

  const line = lines[active];
  const steps = live?.steps ?? line?.steps ?? [];
  const text = live?.text ?? line?.text ?? '';
  const question = live?.question ?? line?.question;
  const certainty = meanTop1(steps);
  const spread = meanEntropy(steps);
  const level = waiting ? 1 : steps.length ? Math.min(0.9, Math.max(0.08, (1 - certainty) * 1.6)) : text ? 0.25 : 0.7;

  useEffect(() => { setSelected((i) => Math.min(i, Math.max(0, steps.length - 1))); }, [steps.length]);
  useEffect(() => {
    if (!autoWalk || !steps.length) return undefined;
    const id = setInterval(() => setSelected((i) => (i + 1) % steps.length), 2000);
    return () => clearInterval(id);
  }, [autoWalk, steps.length]);
  // Keep the newest words in view while the reply streams in.
  useEffect(() => { if (live && replyRef.current) replyRef.current.scrollTop = replyRef.current.scrollHeight; }, [live]);

  const pick = (i) => { setSelected(i); setAutoWalk(false); };

  // One streamed completion. `onPartial` gets { text, steps } as tokens arrive.
  // Returns { acc, stopped }; throws on API errors.
  const complete = async (body, onPartial) => {
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      // The server adds the key and turns logprobs on (see /opt/ideitas/logprob/app.py).
      const res = await fetch('/logprob-gui/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ac.signal,
        body: JSON.stringify({ messages: body, model, temperature: temp }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(t(ERRORS[j.detail] ?? 'errAny'));
      }
      if (!res.body) throw new Error(t('errAny'));
      const acc = await consumeOpenRouterSse(res.body, (p) => {
        partialRef.current = { text: p.text, steps: p.steps };
        onPartial(p);
      }, ac.signal);
      return { acc, stopped: ac.signal.aborted };
    } catch (e) {
      if (e?.name === 'AbortError') return { acc: partialRef.current, stopped: true };
      throw e;
    } finally {
      abortRef.current = null;
    }
  };

  const begin = () => {
    setError('');
    setLatency(null);
    partialRef.current = { text: '', steps: [] };
    setLoading(true);
    screenRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return performance.now();
  };
  const end = () => { setLive(null); setWaiting(false); setLoading(false); };

  const send = async (q = input.trim()) => {
    if (!q || loading) return;
    // The conversation goes on from the future on screen.
    const prev = line?.at != null
      ? [...messages.slice(0, -1), { role: 'assistant', content: line.text }]
      : messages;
    const ctx = [...prev, { role: 'user', content: q }];
    setInput('');
    setMessages(ctx);
    setAutoWalk(true);
    setSelected(0);
    setWaiting(true);
    setLive(null);
    const t0 = begin();
    try {
      const { acc, stopped } = await complete(ctx, (p) => {
        if (p.text || p.steps.length) setWaiting(false);
        setLive({ text: p.text, steps: p.steps, question: q });
      });
      setLatency(Math.round(performance.now() - t0));
      if (acc.text.trim() || acc.steps.length) {
        setLines([{ steps: acc.steps, text: acc.text, question: q, ctx, at: null, alt: null }]);
        setActive(0);
        setMessages([...ctx, { role: 'assistant', content: acc.text.trim() ? acc.text : t('stopped') }]);
        if (!acc.steps.length && !stopped) setError(t('noSteps'));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages(prev);
    } finally {
      end();
    }
  };

  // Open another future: keep the words before step `at`, put `alt` there, let the model go on.
  const branch = async (at, alt, logprob) => {
    if (!line || loading) return;
    const before = line.steps.slice(0, at);
    const head = [...before, { token: alt, logprob, top_logprobs: line.steps[at].top_logprobs }];
    const prefix = head.map((s) => s.token).join('');
    setAutoWalk(false);
    setSelected(at);
    setLive({ steps: head, text: prefix, question: line.question });
    const t0 = begin();
    try {
      const { acc: raw } = await complete(
        [...line.ctx, { role: 'assistant', content: prefix }],
        (p) => { const c = stripEcho(prefix, p); setLive({ steps: [...head, ...c.steps], text: prefix + c.text, question: line.question }); },
      );
      const acc = stripEcho(prefix, raw);
      setLatency(Math.round(performance.now() - t0));
      const next = [...lines, { steps: [...head, ...acc.steps], text: prefix + acc.text, question: line.question, ctx: line.ctx, at, alt }];
      setLines(next);
      setActive(next.length - 1);
      setAutoWalk(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      end();
    }
  };

  const step = steps[selected];
  const show = (tok) => JSON.stringify(tok).slice(1, -1).trim() || '␣';

  return (
    <>
      <div className="bg" aria-hidden="true">
        <Lightfall
          colors={[C.gold, C.pale, '#8a6420']}
          backgroundColor={C.black}
          speed={loading ? 0.7 : 0.22}
          streakCount={3}
          streakWidth={0.6}
          density={0.55}
          glow={0.8}
          zoom={2.6}
          backgroundGlow={0.06}
          headGlow={0.5}
          bloom={0.3}
          mouseInteraction={false}
          holdToWarp={false}
          opacity={0.75}
        />
      </div>

      <div className="room">
        <div className="top">
          <a href="https://ideitas.online/">ideitas.online</a>
          <button onClick={() => setLang(lang === 'es' ? 'en' : 'es')} aria-label="Idioma / Language">{lang === 'es' ? 'EN' : 'ES'}</button>
        </div>

        <header className="hero">
          <h1>Logprob</h1>
          <p>{t('lede')}</p>
        </header>

        {lines.length > 1 && (
          <nav className="lines" aria-label={t('lines')}>
            {lines.map((l, i) => (
              <button
                key={i}
                className={i === active ? 'on' : ''}
                disabled={loading}
                onClick={() => { setActive(i); setSelected(l.at ?? 0); setAutoWalk(l.at == null); }}
              >
                {l.at == null ? t('original') : t('branch', show(l.alt), l.at)}
              </button>
            ))}
          </nav>
        )}

        <section className="screen" ref={screenRef}>
          <Static level={level} storm={waiting} />
          <div className="hud">
            <div className="hud-cell">
              <span>{t('certainty')}</span>
              {certainty == null ? <b>—</b> : <Counter value={Math.round(certainty * 100)} suffix="%" fontSize={22} className="hud-big" />}
            </div>
            <div className="hud-cell"><span>{t('spread')}</span><b>{steps.length ? spread.toFixed(2) : '—'}</b></div>
            <div className="hud-cell"><span>{t('tokens')}</span><b>{steps.length || '—'}</b></div>
            <div className="hud-cell"><span>{t('latency')}</span><b>{latency != null ? `${(latency / 1000).toFixed(1)}s` : '—'}</b></div>
          </div>
          <div className="screen-body" ref={replyRef}>
            {question && <p className="question">{question}</p>}
            {waiting ? (
              <div className="screen-center"><StatusMark status="running" label={t('waiting')} color={C.pale} size={18} fontSize={13} /></div>
            ) : text || steps.length ? (
              <Projection steps={steps} text={text} selected={selected} onSelect={pick} branchAt={live ? null : line?.at} />
            ) : (
              <div className="screen-center">
                <div className="seeds">
                  <span className="idle">{t('idle')}</span>
                  {T[lang].seeds.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="console">
          <div className="ask">
            <textarea
              rows={2}
              value={input}
              placeholder={t('placeholder')}
              spellCheck={false}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return;
                e.preventDefault();
                send();
              }}
            />
            {loading ? (
              <button className="stop" onClick={() => abortRef.current?.abort()}>{t('stop')}</button>
            ) : (
              <SpecularButton
                size="md"
                radius={2}
                textColor={C.pale}
                baseColor={C.dim}
                lineColor={C.gold}
                tint={C.gold}
                tintOpacity={0.05}
                thickness={1}
                glow={0.7}
                intensity={0.9}
                onClick={() => send()}
              >
                {t('send')}
              </SpecularButton>
            )}
          </div>
          {error && <p className="error" role="alert">{error}</p>}
          <div className="settings">
            <GlideSelect
              size="sm"
              radius={2}
              menuWidth={280}
              showTags={false}
              accentColor={C.gold}
              surfaceColor={C.chip}
              highlightColor={C.high}
              textColor={C.ink}
              ariaLabel={t('model')}
              options={MODELS.map((m) => ({ value: m, label: m.split('/')[1] }))}
              value={model}
              onChange={setModel}
            />
            <JellyRadio
              size="sm"
              radius={2}
              gap={4}
              chipColor={C.chip}
              activeColor={temp === 2 ? C.red : C.gold}
              textColor={C.ink}
              activeTextColor={C.black}
              ariaLabel={t('temp')}
              items={TEMPS.map((v, i) => ({ value: v, label: t('temps')[i] }))}
              value={temp}
              onChange={setTemp}
            />
          </div>
        </section>

        {steps.length > 0 && (
          <section className="analysis">
            <div className="block">
              <h2>{t('timeline')}</h2>
              <Timeline steps={steps} selected={selected} onSelect={pick} label={t('timeline')} />
            </div>
            {step && (
              <div className="block">
                <h2>{t('worlds', selected)}</h2>
                <Worlds step={step} onPick={loading ? undefined : (alt, lp) => branch(selected, alt, lp)} />
                <p className="worlds-hint">{t('worldsHint')}</p>
              </div>
            )}
          </section>
        )}

        {messages.length > 0 && (
          <details className="history">
            <summary>{t('history', messages.length)}</summary>
            <ol>
              {messages.map((m, i) => (
                <li key={i} className={m.role}>
                  <b>{m.role === 'user' ? t('you') : t('ai')}</b>
                  <span>{m.content}</span>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </>
  );
}
