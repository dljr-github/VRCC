import React from 'react';
import {Composition} from 'remotion';
import {TOTAL_FRAMES, Walkthrough} from './Walkthrough';
import {FPS} from './theme';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Walkthrough"
      component={Walkthrough}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={1280}
      height={720}
    />
  );
};
