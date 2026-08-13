import type {
  AttentionTask,
  PlaceKey,
  SectionId,
  TimeKey,
} from './state';

/**
 * CONFIGURABLE: below this AI confidence (0–1) the item is flagged
 * "Review required" and does not count as complete until the examiner accepts
 * or overrides the AI result. Confidence is a model/service signal, never a
 * clinical certainty.
 */
export const AI_CONFIDENCE_REVIEW_THRESHOLD = 0.7;

export const MMSE_SECTIONS: { id: SectionId; title: string; max: number }[] = [
  { id: 'orientationTime', title: 'Orientation to Time', max: 5 },
  { id: 'orientationPlace', title: 'Orientation to Place', max: 5 },
  { id: 'registration', title: 'Registration', max: 3 },
  { id: 'attention', title: 'Attention & Calculation', max: 5 },
  { id: 'delayedRecall', title: 'Delayed Recall', max: 3 },
  { id: 'naming', title: 'Naming', max: 2 },
  { id: 'repetition', title: 'Repetition', max: 1 },
  { id: 'command', title: 'Three-Step Command', max: 3 },
  { id: 'reading', title: 'Reading', max: 1 },
  { id: 'writing', title: 'Writing', max: 1 },
  { id: 'copying', title: 'Copying', max: 1 },
];

export const ORIENTATION_TIME_ITEMS: {
  key: TimeKey;
  label: string;
  prompt: string;
}[] = [
  { key: 'year', label: 'Year', prompt: 'What year is it?' },
  { key: 'season', label: 'Season', prompt: 'What season is it?' },
  { key: 'date', label: 'Date', prompt: 'What is the date?' },
  { key: 'day', label: 'Day', prompt: 'What day of the week is it?' },
  { key: 'month', label: 'Month', prompt: 'What month is it?' },
];

export interface PlaceItemConfig {
  key: PlaceKey;
  label: string;
  prompt: string;
  expected: string;
}

/**
 * CONFIGURABLE: correct answers for Orientation to Place depend on the
 * assessment location. Set `expected` for each item at the site where this
 * tool runs. Leave blank if the examiner should rely on their own knowledge;
 * the field is only shown as a hint.
 */
export const PLACE_ITEMS: PlaceItemConfig[] = [
  { key: 'state', label: 'State', prompt: 'What state are we in?', expected: '' },
  { key: 'county', label: 'County', prompt: 'What county are we in?', expected: '' },
  { key: 'town', label: 'Town / City', prompt: 'What town or city are we in?', expected: '' },
  { key: 'building', label: 'Hospital / Building', prompt: 'What hospital or building are we in?', expected: '' },
  { key: 'floor', label: 'Floor', prompt: 'What floor are we on?', expected: '' },
];

/**
 * CONFIGURABLE: the three objects the examiner presents during Registration
 * and later asks the patient to recall during Delayed Recall.
 * These are NOT encoded anywhere else in the project, so they are a
 * configurable default — not claimed to be prescribed by any specific doctor's
 * questionnaire. Replace with your clinic's standard objects if needed.
 */
export const REGISTRATION_OBJECTS: [string, string, string] = [
  'Apple',
  'Table',
  'Penny',
];

export const ATTENTION_TASKS: { id: AttentionTask; label: string }[] = [
  { id: 'serial7', label: 'Serial 7s — subtract 7 from 100 five times' },
  { id: 'spellWorld', label: 'Spell WORLD backwards' },
];

export const SERIAL_7_EXPECTED: string[] = ['93', '86', '79', '72', '65'];

export const SPELL_WORLD_EXPECTED: string[] = ['D', 'L', 'R', 'O', 'W'];

export const NAMING_ITEMS: string[] = ['Wristwatch', 'Pencil'];

export const REPETITION_PHRASE = 'No ifs, ands, or buts.';

export const THREE_STEP_COMMAND_TEXT =
  'Take the paper in your right hand, fold it in half, and put it on the floor.';

export const READING_INSTRUCTION = 'CLOSE YOUR EYES';

export const WRITING_PROMPT = 'Make up and write a sentence about anything.';

/**
 * CONFIGURABLE: path to the exact reference figure from the doctor's MMSE
 * questionnaire used for the Copying section. The asset is NOT bundled with
 * this project, so this is left empty until it is supplied. A placeholder is
 * shown in the Copying section until a path is set here.
 */
export const COPYING_REFERENCE_IMAGE = '';
