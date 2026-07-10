import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('png');
Config.setOverwriteOutput(true);
// The screenshots live in assets/images at the repo root; serving them as
// the public dir means a fresh checkout renders with no copy step.
Config.setPublicDir('../../assets/images');
