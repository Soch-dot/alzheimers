import type { PredictionResponse } from '../api';
import type { AssessmentMode } from './mode';
import { isAssessmentMode } from './mode';
import type { DetailsDraft } from './details';
import { detailsValid, loadStoredDetails } from './details';
import { MMSE_SECTIONS } from './config';
import type { AIScore, CopyingState, ItemState, MMSEState, MmsePhase } from './state';

/**
 * Single source of truth for the in-progress assessment session.
 *
 * ONE versioned localStorage key (`alzheimers_assessment_session_v1`) stores the
 * whole logical position so a browser refresh restores the exact phase (Mode,
 * Assessment Details, MMSE section/summary, Analysis form) plus the MMSE
 * responses, deterministic/AI scores, review/override state, Q6 selection and
 * assessment location.
 *
 * Never persisted: API keys, provider secrets, raw request payloads, and Q11
 * camera/upload images (previewData data URLs are stripped before every save and
 * never restored).
 */
export const SESSION_VERSION = 1;
export const SESSION_KEY = 'alzheimers_assessment_session_v1';
export const LEGACY_DETAILS_KEY = 'alzheimers_assessment_details_v1';

export type AppPhase = 'mode' | 'details' | 'mmse' | 'form';

export interface MmseSessionPart {
  /** Current MMSE step: 0 = intro, 1..11 = sections, 12 = summary. */
  step: number;
  phase: MmsePhase;
  state: MMSEState;
}

export interface AssessmentSession {
  version: number;
  mode: AssessmentMode;
  details: DetailsDraft;
  appPhase: AppPhase;
  mmse: MmseSessionPart | null;
  mmseScore: number | null;
  result: PredictionResponse | null;
}

const MAX_STEP = MMSE_SECTIONS.length + 1;
const MMSE_PHASES: MmsePhase[] = ['collect', 'assessing', 'assessed', 'error'];
const COPYING_STATUSES = ['empty', 'photo', 'analyzing', 'assessed', 'error'];

// ---------------------------------------------------------------------------
// Structural validation. Corrupt / incompatible / stale-version sessions are
// rejected entirely (the caller then clears the key and starts at Mode).
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isScoreMark(value: unknown): value is boolean | null {
  return value === null || typeof value === 'boolean';
}

function isAIScore(value: unknown): value is AIScore {
  return (
    isRecord(value) &&
    typeof value.correct === 'boolean' &&
    typeof value.confidence === 'number' &&
    typeof value.reason === 'string'
  );
}

function isItemState(value: unknown): value is ItemState {
  if (!isRecord(value)) return false;
  return (
    typeof value.response === 'string' &&
    (value.status === 'idle' ||
      value.status === 'assessing' ||
      value.status === 'assessed' ||
      value.status === 'error') &&
    (value.aiScore === null || isAIScore(value.aiScore)) &&
    typeof value.reviewRequired === 'boolean' &&
    typeof value.reviewed === 'boolean' &&
    (value.manual === null || typeof value.manual === 'boolean') &&
    (value.error === null || typeof value.error === 'string')
  );
}

function isItemStateArray(value: unknown, length: number): boolean {
  return Array.isArray(value) && value.length === length && value.every(isItemState);
}

function isStringMap(
  value: unknown,
  keys: string[],
  itemCheck: (v: unknown) => boolean
): boolean {
  if (!isRecord(value)) return false;
  return keys.every((key) => itemCheck(value[key]));
}

function isMMSEState(value: unknown): value is MMSEState {
  if (!isRecord(value)) return false;

  const location = value.location;
  if (!isRecord(location)) return false;
  if (!['state', 'county', 'town', 'building', 'floor'].every((k) => typeof location[k] === 'string')) {
    return false;
  }

  const time = value.orientationTime;
  if (!isRecord(time) || !isStringMap(time.items, ['year', 'season', 'date', 'day', 'month'], isItemState)) {
    return false;
  }

  const place = value.orientationPlace;
  if (!isRecord(place) || !isStringMap(place.items, ['state', 'county', 'town', 'building', 'floor'], isItemState)) {
    return false;
  }

  if (!isItemStateArray((value.registration as Record<string, unknown>)?.items, 3)) return false;

  const attention = value.attention;
  if (!isRecord(attention)) return false;
  if (attention.task === 'serial7') {
    if (!isItemStateArray(attention.serial7, 5)) return false;
  } else if (attention.task === 'spellWorld') {
    const spell = attention.spellWorld;
    if (!isRecord(spell) || typeof spell.response !== 'string') return false;
    if (!isItemStateArray(spell.letters, 5)) return false;
  } else {
    return false;
  }

  if (!isItemStateArray((value.delayedRecall as Record<string, unknown>)?.items, 3)) return false;

  const naming = value.naming;
  if (
    !isRecord(naming) ||
    !isItemState(naming.watch) ||
    !isItemState(naming.pencil) ||
    typeof naming.watchObject !== 'string' ||
    typeof naming.pencilObject !== 'string'
  ) {
    return false;
  }

  if (!isItemState(value.repetition)) return false;

  const command = value.command;
  if (
    !isRecord(command) ||
    !isScoreMark(command.tookPaper) ||
    !isScoreMark(command.foldedPaper) ||
    !isScoreMark(command.placedFloor)
  ) {
    return false;
  }

  const reading = value.reading;
  if (!isRecord(reading) || typeof reading.note !== 'string' || !isScoreMark(reading.correct)) {
    return false;
  }

  if (!isItemState(value.writing)) return false;

  const copying = value.copying;
  if (!isRecord(copying)) return false;
  if (!COPYING_STATUSES.includes(copying.status as string)) return false;
  if (copying.previewData !== null && typeof copying.previewData !== 'string') return false;
  if (typeof copying.previewName !== 'string') return false;
  if (copying.aiScore !== null && !isAIScore(copying.aiScore)) return false;
  if (typeof copying.reviewRequired !== 'boolean') return false;
  if (typeof copying.reviewed !== 'boolean') return false;
  if (copying.manual !== null && typeof copying.manual !== 'boolean') return false;
  if (copying.errorKind !== null && typeof copying.errorKind !== 'string') return false;
  if (copying.errorDetail !== null && typeof copying.errorDetail !== 'string') return false;

  return true;
}

function isDetailsDraftShape(value: unknown): value is DetailsDraft {
  if (!isRecord(value)) return false;
  return (
    typeof value.age === 'string' &&
    (value.sex === '' || value.sex === 0 || value.sex === 1) &&
    typeof value.education_years === 'string' &&
    typeof value.ses === 'string'
  );
}

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!isRecord(value)) return false;
  return (
    typeof value.alzheimers_detected === 'boolean' &&
    typeof value.detection_percentage === 'number' &&
    typeof value.predicted_class === 'string' &&
    typeof value.class_index === 'number' &&
    isRecord(value.probabilities) &&
    typeof value.probabilities.Nondemented === 'number' &&
    typeof value.probabilities.Converted === 'number' &&
    typeof value.probabilities.Demented === 'number'
  );
}

function isValidSession(value: unknown): value is AssessmentSession {
  if (!isRecord(value)) return false;
  if (value.version !== SESSION_VERSION) return false;
  if (!isAssessmentMode(value.mode)) return false;
  if (!isDetailsDraftShape(value.details)) return false;
  if (value.appPhase !== 'mode' && value.appPhase !== 'details' && value.appPhase !== 'mmse' && value.appPhase !== 'form') {
    return false;
  }
  // Reaching the MMSE / Analysis phases requires valid details; a session that
  // claims otherwise is corrupt and is rejected (start over at Mode).
  if (value.appPhase !== 'mode' && value.appPhase !== 'details' && !detailsValid(value.details)) {
    return false;
  }
  if (value.appPhase === 'form' && typeof value.mmseScore !== 'number') return false;
  if (value.mmseScore !== null && typeof value.mmseScore !== 'number') return false;
  if (value.result !== null && value.result !== undefined && !isPredictionResponse(value.result)) {
    return false;
  }
  if (value.mmse !== null && value.mmse !== undefined) {
    const mmse = value.mmse as Record<string, unknown>;
    if (
      typeof mmse.step !== 'number' ||
      !Number.isInteger(mmse.step) ||
      mmse.step < 0 ||
      mmse.step > MAX_STEP
    ) {
      return false;
    }
    if (!MMSE_PHASES.includes(mmse.phase as MmsePhase)) return false;
    if (!isMMSEState(mmse.state)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Q11 image handling. The photo preview is in-memory ONLY — it is stripped on
// save and never restored, so a refresh shows the "Choose/Take Photo" state
// instead of pretending the previous image is still available.
// ---------------------------------------------------------------------------

function sanitizeCopyingForRestore(copying: CopyingState): CopyingState {
  if (copying.status === 'empty') {
    return { ...copying, previewData: null, previewName: '', photoInfo: null };
  }
  // A photo existed only in memory and is gone after refresh. Show the
  // Choose/Take Photo state, but keep any completed AI/manual score so the
  // running total stays accurate. Image + error artifacts are dropped.
  return {
    ...copying,
    status: 'empty',
    previewData: null,
    previewName: '',
    errorKind: null,
    errorDetail: null,
    photoInfo: null,
  };
}

function sanitizeMMSEStateForRestore(state: MMSEState): MMSEState {
  const next = structuredClone(state);
  next.copying = sanitizeCopyingForRestore(next.copying);
  return next;
}

function sanitizeMMSEStateForStorage(state: MMSEState): MMSEState {
  const next = structuredClone(state);
  next.copying = { ...next.copying, previewData: null, previewName: '' };
  return next;
}

function normalizeSession(session: AssessmentSession): AssessmentSession {
  let mmse = session.mmse;
  if (mmse) {
    mmse = {
      ...mmse,
      // 'assessing' / 'error' imply an in-flight or just-failed request whose
      // result is gone after refresh; return to 'collect' so the examiner can
      // re-run the batch instead of being stuck in a dead state.
      phase: mmse.phase === 'assessing' || mmse.phase === 'error' ? 'collect' : mmse.phase,
      state: sanitizeMMSEStateForRestore(mmse.state),
    };
  }
  return { ...session, mmse };
}

export function loadSession(): AssessmentSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) {
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(raw);
      } catch {
        parsed = null;
      }
      if (isValidSession(parsed)) return normalizeSession(parsed);
      // Corrupt / incompatible / stale-version session: clear it and start
      // normally at Assessment Mode. Never restore partial unsafe data.
      localStorage.removeItem(SESSION_KEY);
    }
  } catch {
    // ignore
  }

  // Legacy: the old single `alzheimers_assessment_details_v1` key migrates into
  // the session structure once, then is removed.
  try {
    const legacy = loadStoredDetails();
    if (legacy) {
      localStorage.removeItem(LEGACY_DETAILS_KEY);
      return {
        version: SESSION_VERSION,
        mode: legacy.mode,
        details: {
          age: legacy.age,
          sex: legacy.sex,
          education_years: legacy.education_years,
          ses: legacy.ses,
        },
        appPhase: 'details',
        mmse: null,
        mmseScore: null,
        result: null,
      };
    }
  } catch {
    // ignore
  }
  return null;
}

export function saveSession(session: AssessmentSession): void {
  try {
    const payload: AssessmentSession = {
      ...session,
      mmse: session.mmse
        ? { ...session.mmse, state: sanitizeMMSEStateForStorage(session.mmse.state) }
        : null,
    };
    localStorage.setItem(SESSION_KEY, JSON.stringify(payload));
  } catch {
    // localStorage unavailable (private mode / quota) — non-fatal.
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(LEGACY_DETAILS_KEY);
  } catch {
    // ignore
  }
}
