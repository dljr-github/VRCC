import React from 'react';
import {AbsoluteFill, Series} from 'remotion';
import {EndCard, TitleCard} from './Cards';
import {ScreenSegment, Segment} from './ScreenSegment';
import {END_FRAMES, SEGMENT_FRAMES, TITLE_FRAMES, theme} from './theme';

const segments: Segment[] = [
  {
    step: 1,
    caption:
      'Run VRCC.exe. The wizard picks models for your PC and downloads them.',
    image: 'firstrun.png',
    imgWidth: 709,
    imgHeight: 775,
  },
  {
    step: 2,
    caption:
      'Pick your language and up to three translations, then just talk.',
    image: 'main-window.png',
    imgWidth: 1125,
    imgHeight: 800,
  },
  {
    step: 3,
    caption: 'Tune the voice model and languages any time in Settings.',
    image: 'settings-voice.png',
    imgWidth: 825,
    imgHeight: 905,
  },
  {
    step: 4,
    caption: 'Add or remove models whenever you like.',
    image: 'models.png',
    imgWidth: 850,
    imgHeight: 1125,
  },
];

export const TOTAL_FRAMES =
  TITLE_FRAMES + segments.length * SEGMENT_FRAMES + END_FRAMES;

export const Walkthrough: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: theme.bg}}>
      <Series>
        <Series.Sequence durationInFrames={TITLE_FRAMES}>
          <TitleCard />
        </Series.Sequence>
        {segments.map((s) => (
          <Series.Sequence key={s.step} durationInFrames={SEGMENT_FRAMES}>
            <ScreenSegment {...s} />
          </Series.Sequence>
        ))}
        <Series.Sequence durationInFrames={END_FRAMES}>
          <EndCard />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
