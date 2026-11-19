import { useEffect, useRef } from 'react';

// Gold TV static for the projection screen. `level` (0..1) is how much noise
// covers the screen: 1 before anything is known, low when the model is sure.
export default function Static({ level, storm }) {
  const ref = useRef(null);
  const want = useRef(level);
  const stormRef = useRef(storm);
  want.current = level;
  stormRef.current = storm;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext('2d');
    let w = 0, h = 0, img = null, raf = 0, cur = 1, band = 0;
    const fit = () => {
      // Low resolution on purpose: big soft grain, cheap to fill every frame.
      w = Math.max(1, Math.round(canvas.clientWidth / 3));
      h = Math.max(1, Math.round(canvas.clientHeight / 3));
      canvas.width = w;
      canvas.height = h;
      img = ctx.createImageData(w, h);
    };
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);
    fit();

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const target = stormRef.current ? 1 : want.current;
      cur += (target - cur) * 0.035;
      band = (band + 0.6 + cur * 1.4) % (h + 40);
      const d = img.data;
      for (let y = 0; y < h; y++) {
        const near = Math.abs(y - band + 20);
        const lift = near < 14 ? (1 - near / 14) * 0.5 : 0; // the rolling bright bar
        for (let x = 0; x < w; x++) {
          const i = (y * w + x) * 4;
          const v = Math.random() * 255;
          d[i] = v;
          d[i + 1] = v * 0.8;
          d[i + 2] = v * 0.45;
          d[i + 3] = Math.min(255, (cur * 0.85 + lift * cur) * (110 + Math.random() * 145));
        }
      }
      ctx.putImageData(img, 0, 0);
    };
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={ref} className="static" aria-hidden="true" />;
}
