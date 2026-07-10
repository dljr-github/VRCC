import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {fadeInOut, holdThenFadeOut, settle} from './fades';
import {TITLE_FRAMES, END_FRAMES, theme} from './theme';

const centered: React.CSSProperties = {
  backgroundColor: theme.bg,
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: theme.font,
  flexDirection: 'column',
};

// Fully composed from frame 0 (the poster frame); only the exit animates.
export const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = holdThenFadeOut(frame, TITLE_FRAMES);
  return (
    <AbsoluteFill style={{...centered, opacity}}>
      <div
        style={{
          color: theme.text,
          fontSize: 120,
          fontWeight: 700,
          letterSpacing: 6,
        }}
      >
        VRCC
      </div>
      <div
        style={{
          width: 132,
          height: 5,
          borderRadius: 3,
          backgroundColor: theme.accent,
          margin: '18px 0 30px',
        }}
      />
      <div
        style={{
          color: theme.muted,
          fontSize: 32,
          maxWidth: 1080,
          textAlign: 'center',
        }}
      >
        Your voice, captioned and translated in the VRChat chatbox.
      </div>
    </AbsoluteFill>
  );
};

export const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = fadeInOut(frame, END_FRAMES);
  const rise = (1 - settle(frame, 0, 24)) * 14;
  return (
    <AbsoluteFill style={{...centered, opacity}}>
      <div
        style={{
          color: theme.text,
          fontSize: 44,
          fontWeight: 600,
          transform: `translateY(${rise}px)`,
        }}
      >
        Download the latest release on GitHub.
      </div>
      <div
        style={{
          color: theme.accent,
          fontSize: 30,
          marginTop: 26,
          letterSpacing: 0.5,
          transform: `translateY(${rise}px)`,
        }}
      >
        github.com/dljr-github/VRCC
      </div>
    </AbsoluteFill>
  );
};
