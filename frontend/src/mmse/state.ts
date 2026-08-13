export type ScoreMark = boolean | null;

export type AssessmentStatus = 'idle' | 'assessing' | 'assessed' | 'error';

/**
 * Two-phase assessment workflow.
 * - 'collect'   : patient/examiner records responses only; NO AI calls happen.
 * - 'assessing' : one explicit batch assessment is in flight.
 * - 'assessed'  : batch results received and applied per item.
 * - 'error'     : the batch request failed (provider down / timeout / invalid).
 */
export type MmsePhase = 'collect' | 'assessing' | 'assessed' | 'error';

export interface AIScore {
  correct: boolean;
  confidence: number;
  reason: string;
}

/**
 * Per-item state for AI-scored questions.
 *
 * The patient's response text and the score are kept separate. The score is
 * normally set automatically by the AI service; the examiner can override it
 * (`manual`) or must resolve low-confidence results (`reviewRequired` ->
 * `reviewed`) before the item counts as complete.
 */
export interface ItemState {
  response: string;
  status: AssessmentStatus;
  aiScore: AIScore | null;
  reviewRequired: boolean;
  reviewed: boolean;
  manual: boolean | null;
  error: string | null;
}

export type TimeKey = 'year' | 'season' | 'date' | 'day' | 'month';

export interface OrientationTimeState {
  items: Record<TimeKey, ItemState>;
}

export type PlaceKey = 'state' | 'county' | 'town' | 'building' | 'floor';

/**
 * The assessment location, configured by the examiner before/while running the
 * Orientation to Place section. These values are the reference answers used by
 * the AI during the batch evaluation. They are examiner/AI context only and are
 * never shown to the patient.
 */
export interface AssessmentLocation {
  state: string;
  county: string;
  town: string;
  building: string;
  floor: string;
}

export interface OrientationPlaceState {
  items: Record<PlaceKey, ItemState>;
}

export interface RegistrationState {
  items: [ItemState, ItemState, ItemState];
}

export type AttentionTask = 'serial7' | 'spellWorld';

export interface SpellWorldState {
  response: string;
  letters: [ItemState, ItemState, ItemState, ItemState, ItemState];
}

export interface AttentionState {
  task: AttentionTask;
  serial7: [ItemState, ItemState, ItemState, ItemState, ItemState];
  spellWorld: SpellWorldState;
}

export interface DelayedRecallState {
  items: [ItemState, ItemState, ItemState];
}

export interface NamingState {
  watch: ItemState;
  pencil: ItemState;
}

export interface CommandState {
  tookPaper: ScoreMark;
  foldedPaper: ScoreMark;
  placedFloor: ScoreMark;
}

export interface ReadingState {
  note: string;
  correct: ScoreMark;
}

export type CopyingErrorKind =
  | 'blank'
  | 'timeout'
  | 'unavailable'
  | 'invalid'
  | 'upload';

export type CopyingStatus = 'empty' | 'photo' | 'analyzing' | 'assessed' | 'error';

/**
 * Question 11 (Copying) state — photo-based flow.
 *
 * The patient copies the reference figure on paper; the examiner takes or
 * uploads a photo. The photo preview (`previewData`, an in-memory data URL)
 * exists only for the current assessment and is never written to disk,
 * localStorage, or the backend logs. Analysis runs against the dedicated
 * `/mmse/copying/evaluate` endpoint (independent of the text-MMSE batch).
 */
export interface CopyingState {
  status: CopyingStatus;
  previewData: string | null;
  previewName: string;
  aiScore: AIScore | null;
  reviewRequired: boolean;
  reviewed: boolean;
  manual: boolean | null;
  errorKind: CopyingErrorKind | null;
  errorDetail: string | null;
}

export interface MMSEState {
  location: AssessmentLocation;
  orientationTime: OrientationTimeState;
  orientationPlace: OrientationPlaceState;
  registration: RegistrationState;
  attention: AttentionState;
  delayedRecall: DelayedRecallState;
  naming: NamingState;
  repetition: ItemState;
  command: CommandState;
  reading: ReadingState;
  writing: ItemState;
  copying: CopyingState;
}

export type SectionId =
  | 'orientationTime'
  | 'orientationPlace'
  | 'registration'
  | 'attention'
  | 'delayedRecall'
  | 'naming'
  | 'repetition'
  | 'command'
  | 'reading'
  | 'writing'
  | 'copying';

function blank(): ItemState {
  return {
    response: '',
    status: 'idle',
    aiScore: null,
    reviewRequired: false,
    reviewed: false,
    manual: null,
    error: null,
  };
}

function blankLocation(): AssessmentLocation {
  return { state: '', county: '', town: '', building: '', floor: '' };
}

export function createInitialMMSEState(): MMSEState {
  return {
    location: blankLocation(),
    orientationTime: {
      items: {
        year: blank(),
        season: blank(),
        date: blank(),
        day: blank(),
        month: blank(),
      },
    },
    orientationPlace: {
      items: {
        state: blank(),
        county: blank(),
        town: blank(),
        building: blank(),
        floor: blank(),
      },
    },
    registration: { items: [blank(), blank(), blank()] },
    attention: {
      task: 'serial7',
      serial7: [blank(), blank(), blank(), blank(), blank()],
      spellWorld: {
        response: '',
        letters: [blank(), blank(), blank(), blank(), blank()],
      },
    },
    delayedRecall: { items: [blank(), blank(), blank()] },
    naming: { watch: blank(), pencil: blank() },
    repetition: blank(),
    command: { tookPaper: null, foldedPaper: null, placedFloor: null },
    reading: { note: '', correct: null },
    writing: blank(),
    copying: {
      status: 'empty',
      previewData: null,
      previewName: '',
      aiScore: null,
      reviewRequired: false,
      reviewed: false,
      manual: null,
      errorKind: null,
      errorDetail: null,
    },
  };
}

/** The effective score for an item: examiner manual verdict wins over AI. */
export function effectiveCorrect(item: ItemState): boolean | null {
  if (item.manual !== null) return item.manual;
  return item.aiScore ? item.aiScore.correct : null;
}

/** The effective Q11 score: examiner manual verdict wins over AI. */
export function copyingEffective(copying: CopyingState): boolean | null {
  if (copying.manual !== null) return copying.manual;
  return copying.aiScore ? copying.aiScore.correct : null;
}

/** Whether an item's score is final (counts toward section completion). */
export function isItemFinalized(item: ItemState): boolean {
  if (item.manual !== null) return true;
  if (!item.aiScore) return false;
  return item.reviewRequired ? item.reviewed : true;
}

/** Whether the Q11 score is final (counts toward section completion). */
export function isCopyingFinalized(copying: CopyingState): boolean {
  if (copying.manual !== null) return true;
  if (!copying.aiScore) return false;
  return copying.reviewRequired ? copying.reviewed : true;
}

function countTrue(marks: ScoreMark[]): number {
  return marks.reduce((total, mark) => total + (mark === true ? 1 : 0), 0);
}

export interface MMSEScores {
  orientationTime: number;
  orientationPlace: number;
  registration: number;
  attention: number;
  delayedRecall: number;
  naming: number;
  repetition: number;
  command: number;
  reading: number;
  writing: number;
  copying: number;
}

export function computeScores(state: MMSEState): MMSEScores {
  return {
    orientationTime: countTrue(
      Object.values(state.orientationTime.items).map(effectiveCorrect)
    ),
    orientationPlace: countTrue(
      Object.values(state.orientationPlace.items).map(effectiveCorrect)
    ),
    registration: countTrue(state.registration.items.map(effectiveCorrect)),
    attention: countTrue(
      (
        state.attention.task === 'serial7'
          ? state.attention.serial7
          : state.attention.spellWorld.letters
      ).map(effectiveCorrect)
    ),
    delayedRecall: countTrue(state.delayedRecall.items.map(effectiveCorrect)),
    naming: countTrue([effectiveCorrect(state.naming.watch), effectiveCorrect(state.naming.pencil)]),
    repetition: effectiveCorrect(state.repetition) === true ? 1 : 0,
    command: countTrue([
      state.command.tookPaper,
      state.command.foldedPaper,
      state.command.placedFloor,
    ]),
    reading: state.reading.correct === true ? 1 : 0,
    writing: effectiveCorrect(state.writing) === true ? 1 : 0,
    copying: copyingEffective(state.copying) === true ? 1 : 0,
  };
}

export function computeTotal(state: MMSEState): number {
  const scores = computeScores(state);
  return (
    scores.orientationTime +
    scores.orientationPlace +
    scores.registration +
    scores.attention +
    scores.delayedRecall +
    scores.naming +
    scores.repetition +
    scores.command +
    scores.reading +
    scores.writing +
    scores.copying
  );
}

export function isSectionComplete(id: SectionId, state: MMSEState): boolean {
  switch (id) {
    case 'orientationTime':
      return Object.values(state.orientationTime.items).every(isItemFinalized);
    case 'orientationPlace':
      return Object.values(state.orientationPlace.items).every(isItemFinalized);
    case 'registration':
      return state.registration.items.every(isItemFinalized);
    case 'attention':
      return state.attention.task === 'serial7'
        ? state.attention.serial7.every(isItemFinalized)
        : state.attention.spellWorld.letters.every(isItemFinalized);
    case 'delayedRecall':
      return state.delayedRecall.items.every(isItemFinalized);
    case 'naming':
      return isItemFinalized(state.naming.watch) && isItemFinalized(state.naming.pencil);
    case 'repetition':
      return isItemFinalized(state.repetition);
    case 'command':
      return (
        state.command.tookPaper !== null &&
        state.command.foldedPaper !== null &&
        state.command.placedFloor !== null
      );
    case 'reading':
      return state.reading.correct !== null;
    case 'writing':
      return isItemFinalized(state.writing);
    case 'copying':
      return isCopyingFinalized(state.copying);
  }
}