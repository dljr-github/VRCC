import {interpolate} from 'remotion';

// Scenes settle into a static hold quickly; long unchanging runs keep the
// GIF small because encoders store them as tiny frame diffs.
export const fadeInOut = (frame: number, duration: number): number =>
  interpolate(frame, [0, 10, duration - 10, duration], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

// GitHub uses frame 0 as the README's poster image, and viewers with
// animation paused or reduced motion never see past it, so the opening
// scene must hold full opacity from its first frame and only fade out.
export const holdThenFadeOut = (frame: number, duration: number): number =>
  interpolate(frame, [duration - 10, duration], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

export const settle = (frame: number, delay: number, length: number): number =>
  interpolate(frame, [delay, delay + length], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: (t) => 1 - (1 - t) * (1 - t) * (1 - t),
  });
