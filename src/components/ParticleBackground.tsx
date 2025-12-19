import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const MAX_DISTANCE = 150;
const PARTICLE_RADIUS = 2;
const MASK_SELECTOR = "[data-particle-mask=\"off\"]";

interface Props {
  count?: number;
}

const ParticleBackground = ({ count = 90 }: Props) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;

    const createParticle = (width: number, height: number): Particle => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
    });

    const particles: Particle[] = Array.from({ length: count }, () => ({
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
    }));
    let maskRects: DOMRect[] = [];

    const updateMaskRects = () => {
      maskRects = Array.from(document.querySelectorAll(MASK_SELECTOR)).map((el) => el.getBoundingClientRect());
    };

    const syncParticlesToCanvas = () => {
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      particles.forEach((particle, index) => {
        if (particle.vx === 0 && particle.vy === 0 && particle.x === 0 && particle.y === 0) {
          const seed = createParticle(width, height);
          particles[index] = seed;
          return;
        }
        particle.x = Math.min(Math.max(particle.x, 0), width);
        particle.y = Math.min(Math.max(particle.y, 0), height);
      });
    };

    const resize = () => {
      const { innerWidth, innerHeight } = window;
      canvas.width = innerWidth * dpr;
      canvas.height = innerHeight * dpr;
      canvas.style.width = `${innerWidth}px`;
      canvas.style.height = `${innerHeight}px`;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
      syncParticlesToCanvas();
      updateMaskRects();
    };

    resize();
    updateMaskRects();

    const mouse = { x: 0, y: 0, active: false };

    const handleMouseMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = event.clientX - rect.left;
      mouse.y = event.clientY - rect.top;
      mouse.active = true;
    };

    const handleMouseLeave = () => {
      mouse.active = false;
    };

    let scrollRaf = 0;
    const handleScroll = () => {
      if (scrollRaf) return;
      scrollRaf = window.requestAnimationFrame(() => {
        updateMaskRects();
        scrollRaf = 0;
      });
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseleave", handleMouseLeave);
    window.addEventListener("scroll", handleScroll, { passive: true });

    let animationFrameId = 0;

    const draw = () => {
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      ctx.clearRect(0, 0, width, height);

      particles.forEach((particle, index) => {
        particle.x += particle.vx;
        particle.y += particle.vy;

        if (particle.x <= 0 || particle.x >= width) {
          particle.vx *= -1;
          particle.x = Math.min(Math.max(particle.x, 0), width);
        }
        if (particle.y <= 0 || particle.y >= height) {
          particle.vy *= -1;
          particle.y = Math.min(Math.max(particle.y, 0), height);
        }

        const isMasked = maskRects.some((rect) =>
          particle.x >= rect.left && particle.x <= rect.right && particle.y >= rect.top && particle.y <= rect.bottom,
        );
        if (isMasked) {
          return;
        }

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, PARTICLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(99, 102, 241, 0.65)";
        ctx.fill();

        for (let j = index + 1; j < particles.length; j++) {
          const other = particles[j];
          const otherMasked = maskRects.some((rect) =>
            other.x >= rect.left && other.x <= rect.right && other.y >= rect.top && other.y <= rect.bottom,
          );
          if (otherMasked) continue;

          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.hypot(dx, dy);

          if (distance < MAX_DISTANCE) {
            const opacity = 1 - distance / MAX_DISTANCE;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(129, 140, 248, ${opacity * 0.8})`;
            ctx.lineWidth = 0.8;
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            ctx.stroke();
          }
        }

        if (mouse.active) {
          const dx = particle.x - mouse.x;
          const dy = particle.y - mouse.y;
          const distance = Math.hypot(dx, dy);
          if (distance < MAX_DISTANCE) {
            const opacity = 1 - distance / MAX_DISTANCE;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(165, 180, 252, ${opacity * 0.9})`;
            ctx.lineWidth = 0.8;
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(mouse.x, mouse.y);
            ctx.stroke();
          }
        }
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (scrollRaf) cancelAnimationFrame(scrollRaf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseleave", handleMouseLeave);
      window.removeEventListener("scroll", handleScroll);
    };
  }, [count]);

  return <canvas ref={canvasRef} className="pointer-events-none fixed inset-0 -z-10 h-full w-full" />;
};

export default ParticleBackground;
