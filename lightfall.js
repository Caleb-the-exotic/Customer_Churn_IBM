/**
 * Lightfall Full-Screen Animation Engine
 * Creates a luminous full-screen cascade of glowing light beams and starfall streaks
 * matching @react-bits/Lightfall-JS-TW specifications:
 *   - colors: ['#A6C8FF', '#5227FF', '#FF9FFC']
 *   - backgroundColor: '#0A29FF'
 *   - speed: 0.5, streakCount: 2, streakWidth: 1, streakLength: 1, glow: 1, density: 0.6, twinkle: 1, zoom: 3
 *   - mouseInteraction: true, mouseStrength: 0.5, mouseRadius: 1
 */

(function (global) {
  function initLightfall(container, options = {}) {
    if (!container) return null;

    // Clear previous elements
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const opts = Object.assign({
      colors: ['#A6C8FF', '#5227FF', '#FF9FFC'],
      backgroundColor: '#0A29FF',
      speed: 0.15,
      streakCount: 2,
      streakWidth: 1.2,
      streakLength: 1.1,
      glow: 1.1,
      density: 0.6,
      twinkle: 0.8,
      zoom: 3.0,
      backgroundGlow: 0.5,
      opacity: 1.0,
      mouseInteraction: true,
      mouseStrength: 0.35,
      mouseRadius: 1.0
    }, options);

    const canvas = document.createElement('canvas');
    canvas.id = 'lightfall-canvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvas.style.pointerEvents = 'none';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Simulation entities
    let width = 0;
    let height = 0;
    let dpr = 1;

    let mouseX = -9999;
    let mouseY = -9999;
    let targetMouseX = -9999;
    let targetMouseY = -9999;

    // Number of streaks based on density
    const numStreaks = Math.round(45 * opts.density);
    const numStars = Math.round(160 * opts.density);

    // Beams of lightfall
    const streaks = [];
    for (let i = 0; i < numStreaks; i++) {
      streaks.push(createStreak(true));
    }

    // Twinkling stars
    const stars = [];
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random(),
        y: Math.random(),
        radius: Math.random() * 1.8 + 0.4,
        color: opts.colors[Math.floor(Math.random() * opts.colors.length)],
        twinkleSpeed: Math.random() * 1.5 + 0.5,
        twinkleOffset: Math.random() * Math.PI * 2,
        baseAlpha: Math.random() * 0.5 + 0.25
      });
    }

    function createStreak(randomY = false) {
      const colorIdx = Math.floor(Math.random() * opts.colors.length);
      const angle = (72 + (Math.random() - 0.5) * 10) * (Math.PI / 180); // ~72 deg downward
      const baseLen = (130 + Math.random() * 170) * opts.streakLength;
      // Reduced speed for gentle, majestic starfall motion across full screen
      const baseSpeed = (0.07 + Math.random() * 0.11) * (opts.speed / 0.15);

      return {
        x: Math.random() * 1.3 - 0.15, // normalized 0..1
        y: randomY ? Math.random() : -0.15,
        length: baseLen,
        speed: baseSpeed,
        width: (1.5 + Math.random() * 2.2) * opts.streakWidth,
        color: opts.colors[colorIdx],
        secondaryColor: opts.colors[(colorIdx + 1) % opts.colors.length],
        angle: angle,
        alpha: Math.random() * 0.4 + 0.6,
        sparks: []
      };
    }

    function resize() {
      const rect = container.getBoundingClientRect();
      width = Math.max(rect.width || window.innerWidth, 320);
      height = Math.max(rect.height || window.innerHeight, 320);
      dpr = Math.min(window.devicePixelRatio || 1, 2);

      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    window.addEventListener('resize', resize, { passive: true });

    function onMouseMove(e) {
      targetMouseX = e.clientX;
      targetMouseY = e.clientY;
    }

    if (opts.mouseInteraction) {
      window.addEventListener('pointermove', onMouseMove, { passive: true });
    }

    let animId = null;
    let lastTime = performance.now();

    function render(now) {
      animId = requestAnimationFrame(render);
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      // Mouse smoothing
      mouseX += (targetMouseX - mouseX) * 0.12;
      mouseY += (targetMouseY - mouseY) * 0.12;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Deep vibrant background
      ctx.fillStyle = '#040818';
      ctx.fillRect(0, 0, width, height);

      // Cosmic glow core
      const bgRadial = ctx.createRadialGradient(
        width * 0.5, height * 0.35, 20,
        width * 0.5, height * 0.45, Math.max(width, height) * 0.8
      );
      bgRadial.addColorStop(0, 'rgba(10, 41, 255, 0.35)');
      bgRadial.addColorStop(0.35, 'rgba(82, 39, 255, 0.22)');
      bgRadial.addColorStop(0.7, 'rgba(12, 20, 60, 0.4)');
      bgRadial.addColorStop(1, 'rgba(4, 8, 24, 0.95)');
      ctx.fillStyle = bgRadial;
      ctx.fillRect(0, 0, width, height);

      // Dynamic mouse interactive light dome
      if (opts.mouseInteraction && mouseX > 0 && mouseY > 0) {
        const mouseR = 260 * opts.mouseRadius;
        const mouseGrad = ctx.createRadialGradient(mouseX, mouseY, 0, mouseX, mouseY, mouseR);
        mouseGrad.addColorStop(0, `rgba(166, 200, 255, ${0.28 * opts.mouseStrength})`);
        mouseGrad.addColorStop(0.4, `rgba(82, 39, 255, ${0.18 * opts.mouseStrength})`);
        mouseGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = mouseGrad;
        ctx.fillRect(0, 0, width, height);
      }

      // Render Twinkling Stars
      const timeSec = now * 0.001;
      stars.forEach(s => {
        const twinkleFactor = Math.sin(timeSec * s.twinkleSpeed + s.twinkleOffset) * opts.twinkle;
        const alpha = Math.max(0.08, Math.min(1.0, s.baseAlpha + twinkleFactor * 0.35));

        ctx.fillStyle = s.color;
        ctx.globalAlpha = alpha * opts.opacity;
        ctx.beginPath();
        ctx.arc(s.x * width, s.y * height, s.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      // Render Lightfall Streaks
      ctx.globalAlpha = opts.opacity;
      streaks.forEach(st => {
        // Move streak
        let currentSpeed = st.speed;
        let px = st.x * width;
        let py = st.y * height;

        // Mouse deflection & speed boost
        if (opts.mouseInteraction && mouseX > 0) {
          const dx = px - mouseX;
          const dy = py - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const reach = 220 * opts.mouseRadius;
          if (dist < reach) {
            const force = (1 - dist / reach) * opts.mouseStrength;
            currentSpeed += force * 0.25;
            st.x += (dx / dist) * force * 0.003;
          }
        }

        st.y += (currentSpeed * dt);
        st.x += (Math.cos(st.angle) * currentSpeed * dt * 0.35);

        // Reset if offscreen
        if (st.y > 1.25 || st.x > 1.25 || st.x < -0.25) {
          Object.assign(st, createStreak(false));
          px = st.x * width;
          py = st.y * height;
        }

        // Streak geometry
        const endX = px - Math.cos(st.angle) * st.length;
        const endY = py - Math.sin(st.angle) * st.length;

        // Gradient for lightfall beam
        const grad = ctx.createLinearGradient(px, py, endX, endY);
        grad.addColorStop(0, '#FFFFFF');
        grad.addColorStop(0.12, st.color);
        grad.addColorStop(0.65, st.secondaryColor);
        grad.addColorStop(1, 'rgba(10, 41, 255, 0)');

        // Draw beam
        ctx.strokeStyle = grad;
        ctx.lineWidth = st.width;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Glowing head bulb
        const headGlow = ctx.createRadialGradient(px, py, 0, px, py, st.width * 4.5 * opts.glow);
        headGlow.addColorStop(0, '#FFFFFF');
        headGlow.addColorStop(0.3, st.color);
        headGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = headGlow;
        ctx.beginPath();
        ctx.arc(px, py, st.width * 4.5 * opts.glow, 0, Math.PI * 2);
        ctx.fill();

        // Trailing sparks
        if (Math.random() < 0.22) {
          st.sparks.push({
            x: px + (Math.random() - 0.5) * 6,
            y: py + (Math.random() - 0.5) * 6,
            life: 1.0,
            decay: Math.random() * 1.5 + 1.2,
            size: Math.random() * 1.8 + 0.8,
            color: st.color
          });
        }

        // Draw and update sparks
        for (let j = st.sparks.length - 1; j >= 0; j--) {
          const sp = st.sparks[j];
          sp.life -= dt * sp.decay;
          if (sp.life <= 0) {
            st.sparks.splice(j, 1);
            continue;
          }
          ctx.fillStyle = sp.color;
          ctx.globalAlpha = sp.life * opts.opacity * 0.8;
          ctx.beginPath();
          ctx.arc(sp.x, sp.y, sp.size * sp.life, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      ctx.restore();
    }

    animId = requestAnimationFrame(render);

    return {
      resize: resize,
      destroy: function () {
        if (animId) cancelAnimationFrame(animId);
        window.removeEventListener('pointermove', onMouseMove);
        window.removeEventListener('resize', resize);
        ro.disconnect();
        if (canvas.parentElement === container) {
          container.removeChild(canvas);
        }
      }
    };
  }

  global.Lightfall = {
    init: initLightfall
  };
})(window);
