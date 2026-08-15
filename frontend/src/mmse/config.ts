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
}

/**
 * The five Orientation to Place questions. The reference answers are NOT
 * hardcoded here — they are configured by the examiner at assessment time via
 * the "Assessment location" form (see `LOCATION_FIELDS` below) and are used
 * only as examiner/AI context during the batch evaluation. They are never shown
 * to the patient.
 */
export const PLACE_ITEMS: PlaceItemConfig[] = [
  { key: 'state', label: 'State', prompt: 'What state are we in?' },
  { key: 'county', label: 'County / District', prompt: 'What county or district are we in?' },
  { key: 'town', label: 'Town / City', prompt: 'What town or city are we in?' },
  { key: 'building', label: 'Hospital / Building', prompt: 'What hospital or building are we in?' },
  { key: 'floor', label: 'Floor', prompt: 'What floor are we on?' },
];

/**
 * CONFIGURABLE: the assessment-location form shown to the examiner at the top
 * of the Orientation to Place section. The values entered here become the
 * reference answers the AI uses for the five items. Patient-facing UI never
 * shows these values.
 */
export const LOCATION_FIELDS: { key: PlaceKey; label: string; placeholder: string }[] = [
  { key: 'state', label: 'State', placeholder: 'e.g. California' },
  { key: 'county', label: 'County / District', placeholder: 'e.g. Los Angeles County' },
  { key: 'town', label: 'Town / City', placeholder: 'e.g. San Francisco' },
  { key: 'building', label: 'Hospital / Building', placeholder: 'e.g. General Hospital' },
  { key: 'floor', label: 'Floor', placeholder: 'e.g. 2' },
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

export const REPETITION_PHRASE = 'No ifs, ands, or buts.';

export const THREE_STEP_COMMAND_TEXT =
  'Take the paper in your right hand, fold it in half, and put it on the floor.';

export const READING_INSTRUCTION = 'CLOSE YOUR EYES';

export const WRITING_PROMPT = 'Make up and write a sentence about anything.';

/**
 * Path (from `frontend/public/`) to the exact reference figure from the doctor's
 * MMSE questionnaire used for the Copying section. The asset is served from
 * `frontend/public/mmse-copying-figure.png` and is the ONLY stimulus shown to
 * the patient. Do not substitute or redraw it.
 */
export const COPYING_REFERENCE_IMAGE = '/mmse-copying-figure.png';
