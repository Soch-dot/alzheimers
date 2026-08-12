export type ScoreMark = boolean | null;

export interface OrientationTimeState {
  year: ScoreMark;
  season: ScoreMark;
  date: ScoreMark;
  day: ScoreMark;
  month: ScoreMark;
}

export type PlaceKey = 'state' | 'county' | 'town' | 'building' | 'floor';

export interface PlaceItemState {
  response: string;
  correct: ScoreMark;
}

export interface OrientationPlaceState {
  items: Record<PlaceKey, PlaceItemState>;
}

export interface RegistrationState {
  recalled: [ScoreMark, ScoreMark, ScoreMark];
}

export type AttentionTask = 'serial7' | 'spellWorld';

export interface AttentionState {
  task: AttentionTask;
  serial7: [ScoreMark, ScoreMark, ScoreMark, ScoreMark, ScoreMark];
  spellWorld: [ScoreMark, ScoreMark, ScoreMark, ScoreMark, ScoreMark];
}

export interface DelayedRecallState {
  recalled: [ScoreMark, ScoreMark, ScoreMark];
}

export interface NamingState {
  watch: ScoreMark;
  pencil: ScoreMark;
}

export interface CommandState {
  tookPaper: ScoreMark;
  foldedPaper: ScoreMark;
  placedFloor: ScoreMark;
}

export interface WritingState {
  response: string;
  correct: ScoreMark;
}

export interface MMSEState {
  orientationTime: OrientationTimeState;
  orientationPlace: OrientationPlaceState;
  registration: RegistrationState;
  attention: AttentionState;
  delayedRecall: DelayedRecallState;
  naming: NamingState;
  repetition: ScoreMark;
  command: CommandState;
  reading: ScoreMark;
  writing: WritingState;
  copying: ScoreMark;
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

export function createInitialMMSEState(): MMSEState {
  return {
    orientationTime: {
      year: null,
      season: null,
      date: null,
      day: null,
      month: null,
    },
    orientationPlace: {
      items: {
        state: { response: '', correct: null },
        county: { response: '', correct: null },
        town: { response: '', correct: null },
        building: { response: '', correct: null },
        floor: { response: '', correct: null },
      },
    },
    registration: { recalled: [null, null, null] },
    attention: {
      task: 'serial7',
      serial7: [null, null, null, null, null],
      spellWorld: [null, null, null, null, null],
    },
    delayedRecall: { recalled: [null, null, null] },
    naming: { watch: null, pencil: null },
    repetition: null,
    command: { tookPaper: null, foldedPaper: null, placedFloor: null },
    reading: null,
    writing: { response: '', correct: null },
    copying: null,
  };
}

function count(marks: ScoreMark[]): number {
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
    orientationTime: count(Object.values(state.orientationTime)),
    orientationPlace: count(
      Object.values(state.orientationPlace.items).map((item) => item.correct)
    ),
    registration: count([...state.registration.recalled]),
    attention: count(
      state.attention.task === 'serial7'
        ? [...state.attention.serial7]
        : [...state.attention.spellWorld]
    ),
    delayedRecall: count([...state.delayedRecall.recalled]),
    naming: count([state.naming.watch, state.naming.pencil]),
    repetition: state.repetition === true ? 1 : 0,
    command: count([
      state.command.tookPaper,
      state.command.foldedPaper,
      state.command.placedFloor,
    ]),
    reading: state.reading === true ? 1 : 0,
    writing: state.writing.correct === true ? 1 : 0,
    copying: state.copying === true ? 1 : 0,
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
      return Object.values(state.orientationTime).every((mark) => mark !== null);
    case 'orientationPlace':
      return Object.values(state.orientationPlace.items).every(
        (item) => item.correct !== null
      );
    case 'registration':
      return state.registration.recalled.every((mark) => mark !== null);
    case 'attention':
      return state.attention.task === 'serial7'
        ? state.attention.serial7.every((mark) => mark !== null)
        : state.attention.spellWorld.every((mark) => mark !== null);
    case 'delayedRecall':
      return state.delayedRecall.recalled.every((mark) => mark !== null);
    case 'naming':
      return state.naming.watch !== null && state.naming.pencil !== null;
    case 'repetition':
      return state.repetition !== null;
    case 'command':
      return (
        state.command.tookPaper !== null &&
        state.command.foldedPaper !== null &&
        state.command.placedFloor !== null
      );
    case 'reading':
      return state.reading !== null;
    case 'writing':
      return state.writing.correct !== null;
    case 'copying':
      return state.copying !== null;
  }
}
