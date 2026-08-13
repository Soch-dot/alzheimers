export type ScoreMark = boolean | null;

/**
 * Patient responses are stored separately from examiner scores:
 * `response` records the patient's actual answer text (never auto-scored),
 * `correct` is the examiner's independent Correct/Incorrect mark.
 */
export interface ItemState {
  response: string;
  correct: ScoreMark;
}

export type TimeKey = 'year' | 'season' | 'date' | 'day' | 'month';

export interface OrientationTimeState {
  items: Record<TimeKey, ItemState>;
}

export type PlaceKey = 'state' | 'county' | 'town' | 'building' | 'floor';

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

export interface MMSEState {
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

function blank(): ItemState {
  return { response: '', correct: null };
}

export function createInitialMMSEState(): MMSEState {
  return {
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
    orientationTime: count(
      Object.values(state.orientationTime.items).map((item) => item.correct)
    ),
    orientationPlace: count(
      Object.values(state.orientationPlace.items).map((item) => item.correct)
    ),
    registration: count(state.registration.items.map((item) => item.correct)),
    attention: count(
      (
        state.attention.task === 'serial7'
          ? state.attention.serial7
          : state.attention.spellWorld.letters
      ).map((item) => item.correct)
    ),
    delayedRecall: count(state.delayedRecall.items.map((item) => item.correct)),
    naming: count([state.naming.watch.correct, state.naming.pencil.correct]),
    repetition: state.repetition.correct === true ? 1 : 0,
    command: count([
      state.command.tookPaper,
      state.command.foldedPaper,
      state.command.placedFloor,
    ]),
    reading: state.reading.correct === true ? 1 : 0,
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
      return Object.values(state.orientationTime.items).every(
        (item) => item.correct !== null
      );
    case 'orientationPlace':
      return Object.values(state.orientationPlace.items).every(
        (item) => item.correct !== null
      );
    case 'registration':
      return state.registration.items.every((item) => item.correct !== null);
    case 'attention':
      return state.attention.task === 'serial7'
        ? state.attention.serial7.every((item) => item.correct !== null)
        : state.attention.spellWorld.letters.every(
            (item) => item.correct !== null
          );
    case 'delayedRecall':
      return state.delayedRecall.items.every((item) => item.correct !== null);
    case 'naming':
      return (
        state.naming.watch.correct !== null &&
        state.naming.pencil.correct !== null
      );
    case 'repetition':
      return state.repetition.correct !== null;
    case 'command':
      return (
        state.command.tookPaper !== null &&
        state.command.foldedPaper !== null &&
        state.command.placedFloor !== null
      );
    case 'reading':
      return state.reading.correct !== null;
    case 'writing':
      return state.writing.correct !== null;
    case 'copying':
      return state.copying !== null;
  }
}
