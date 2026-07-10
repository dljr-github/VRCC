import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {fadeInOut, settle} from './fades';
import {SEGMENT_FRAMES, theme} from './theme';

export type Segment = {
  step: number;
  caption: string;
  image: string;
  imgWidth: number;
  imgHeight: number;
};

// The frame is 1280x720. Text sits in a fixed left column so the eye
// returns to the same spot each step; the screenshot fills the rest.
const IMG_BOX_W = 680;
const IMG_BOX_H = 620;

export const ScreenSegment: React.FC<Segment> = ({
  step,
  caption,
  image,
  imgWidth,
  imgHeight,
}) => {
  const frame = useCurrentFrame();
  const opacity = fadeInOut(frame, SEGMENT_FRAMES);
  const enter = settle(frame, 4, 26);
  const textEnter = settle(frame, 10, 22);

  const fit = Math.min(IMG_BOX_W / imgWidth, IMG_BOX_H / imgHeight);
  const w = Math.round(imgWidth * fit);
  const h = Math.round(imgHeight * fit);

  return (
    <AbsoluteFill
      style={{
        backgroundColor: theme.bg,
        fontFamily: theme.font,
        flexDirection: 'row',
        alignItems: 'center',
        padding: '0 56px',
        opacity,
      }}
    >
      <div
        style={{
          width: 430,
          flexShrink: 0,
          paddingRight: 48,
          opacity: textEnter,
          transform: `translateY(${(1 - textEnter) * 12}px)`,
        }}
      >
        <div
          style={{
            color: theme.accent,
            fontSize: 22,
            fontWeight: 600,
            letterSpacing: 3,
            textTransform: 'uppercase',
            marginBottom: 18,
          }}
        >
          Step {step} of 4
        </div>
        <div
          style={{
            color: theme.text,
            fontSize: 36,
            fontWeight: 400,
            lineHeight: 1.4,
          }}
        >
          {caption}
        </div>
      </div>
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Img
          src={staticFile(image)}
          style={{
            width: w,
            height: h,
            borderRadius: 10,
            border: `1px solid #2a2f3d`,
            boxShadow: '0 18px 48px rgba(0, 0, 0, 0.45)',
            transform: `scale(${0.965 + 0.035 * enter})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
