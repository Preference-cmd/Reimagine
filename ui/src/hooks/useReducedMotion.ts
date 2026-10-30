import { useEffect, useState } from "react";

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function reducedMotionQuery(): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return null;
  return window.matchMedia(REDUCED_MOTION_QUERY);
}

/**
 * Tracks the OS "reduce motion" preference so inline transitions can be
 * dropped. Returns false where `matchMedia` is unavailable (tests, SSR).
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => reducedMotionQuery()?.matches ?? false);

  useEffect(() => {
    const media = reducedMotionQuery();
    if (!media) return;
    const onChange = () => setReduced(media.matches);
    // Older engines and partial test stubs only expose one listener API.
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onChange);
      return () => media.removeEventListener("change", onChange);
    }
    if (typeof media.addListener === "function") {
      media.addListener(onChange);
      return () => media.removeListener(onChange);
    }
  }, []);

  return reduced;
}
