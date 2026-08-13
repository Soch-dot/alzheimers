import type { ItemState, MMSEState, PlaceKey, SectionId, TimeKey } from './state';
import {
  AI_CONFIDENCE_REVIEW_THRESHOLD,
  ORIENTATION_TIME_ITEMS,
  PLACE_ITEMS,
  REGISTRATION_OBJECTS,
  REPETITION_PHRASE,
  SERIAL_7_EXPECTED,
  SPELL_WORLD_EXPECTED,
  WRITING_PROMPT,
} from './config';
import type { MmseBatchItem, MmseBatchResponse } from '../api';

/**
 * Batch-assessment helpers.
 *
 * Phase 1 (collect) only records responses — no AI. Phase 2 sends ONE batch
 * request containing every applicable, answered response. This module maps
 * MMSEState <-> the backend batch payload and applies per-item results back.
 */

/**
 * Collect the responses eligible for AI evaluation.
 * `onlyUnscored` skips items that already have an examiner manual verdict or an
 * AI score (used to re-assess just what is still missing after edits).
 * Items with empty responses are never sent, and orientation-place items whose
 * expected answer is not configured stay manual-only.
 */
export function buildBatchItems(
  state: MMSEState,
  onlyUnscored = false
): Record<string, MmseBatchItem> {
  const items: Record<string, MmseBatchItem> = {};

  const add = (key: string, question: string, item: ItemState, expected: string) => {
    const response = (item.response ?? '').trim();
    if (!response) return;
    if (onlyUnscored && (item.manual !== null || item.aiScore !== null)) return;
    items[key] = { question, response, expected };
  };

  ORIENTATION_TIME_ITEMS.forEach((it) =>
    add(`orientation_time.${it.key}`, it.prompt, state.orientationTime.items[it.key], '')
  );

  PLACE_ITEMS.forEach((it) => {
    if (!it.expected) return; // manual-only when expected not configured
    add(`orientation_place.${it.key}`, it.prompt, state.orientationPlace.items[it.key], it.expected);
  });

  state.registration.items.forEach((item, i) =>
    add(`registration.${i + 1}`, `Object ${i + 1}`, item, REGISTRATION_OBJECTS[i])
  );

  if (state.attention.task === 'serial7') {
    state.attention.serial7.forEach((item, i) =>
      add(`attention_serial7.${i + 1}`, `Subtraction ${i + 1}`, item, SERIAL_7_EXPECTED[i] ?? '')
    );
  } else {
    state.attention.spellWorld.letters.forEach((item, i) =>
      add(`attention_spell_world.${i + 1}`, `Letter ${i + 1} of WORLD backwards`, item, SPELL_WORLD_EXPECTED[i])
    );
  }

  state.delayedRecall.items.forEach((item, i) =>
    add(`delayed_recall.${i + 1}`, `Response ${i + 1}`, item, REGISTRATION_OBJECTS[i])
  );

  add('naming.wristwatch', 'What is this?', state.naming.watch, 'wristwatch');
  add('naming.pencil', 'What is this?', state.naming.pencil, 'pencil');
  add('repetition.phrase', 'Repeat the required phrase', state.repetition, REPETITION_PHRASE);
  add('writing.sentence', WRITING_PROMPT, state.writing, '');

  return items;
}

function findItem(state: MMSEState, key: string): ItemState | undefined {
  const [section, itemKey] = key.split('.');
  switch (section) {
    case 'orientation_time':
      return state.orientationTime.items[itemKey as TimeKey];
    case 'orientation_place':
      return state.orientationPlace.items[itemKey as PlaceKey];
    case 'registration':
      return state.registration.items[Number(itemKey) - 1];
    case 'attention_serial7':
      return state.attention.serial7[Number(itemKey) - 1];
    case 'attention_spell_world':
      return state.attention.spellWorld.letters[Number(itemKey) - 1];
    case 'delayed_recall':
      return state.delayedRecall.items[Number(itemKey) - 1];
    case 'naming':
      return itemKey === 'wristwatch' ? state.naming.watch : state.naming.pencil;
    case 'repetition':
      return state.repetition;
    case 'writing':
      return state.writing;
    default:
      return undefined;
  }
}

/**
 * Apply a batch outcome (per-item results + per-item errors) onto a draft of
 * the MMSE state. Items with a valid result get their AI score; items that
 * failed validation are marked `error` (no score is silently assigned).
 */
export function applyBatchResultsToDraft(
  draft: MMSEState,
  outcome: MmseBatchResponse
): void {
  const keys = new Set([
    ...Object.keys(outcome.items),
    ...Object.keys(outcome.errors ?? {}),
  ]);

  keys.forEach((key) => {
    const item = findItem(draft, key);
    if (!item) return;

    const res = outcome.items[key];
    if (res) {
      item.status = 'assessed';
      item.aiScore = {
        correct: res.correct,
        confidence: res.confidence,
        reason: res.reason,
      };
      item.reviewRequired = res.confidence < AI_CONFIDENCE_REVIEW_THRESHOLD;
      item.reviewed = false;
      item.error = null;
    } else if (outcome.errors?.[key]) {
      item.status = 'error';
      item.error = outcome.errors[key];
    }
  });
}

export interface SectionResponseCount {
  done: number;
  max: number;
}

/**
 * How many required entries each section has collected (responses for AI items,
 * examiner marks for manual/visual items). Used during Phase 1 instead of
 * scores, which are only meaningful after AI assessment.
 */
export function sectionResponseCounts(
  state: MMSEState
): Record<SectionId, SectionResponseCount> {
  const nonEmpty = (item: ItemState) => (item.response ?? '').trim() !== '';

  const counts: Record<SectionId, number> = {
    orientationTime: Object.values(state.orientationTime.items).filter(nonEmpty).length,
    orientationPlace: PLACE_ITEMS.filter((it) =>
      it.expected
        ? nonEmpty(state.orientationPlace.items[it.key])
        : state.orientationPlace.items[it.key].manual !== null
    ).length,
    registration: state.registration.items.filter(nonEmpty).length,
    attention:
      state.attention.task === 'serial7'
        ? state.attention.serial7.filter(nonEmpty).length
        : state.attention.spellWorld.response.trim()
          ? 5
          : 0,
    delayedRecall: state.delayedRecall.items.filter(nonEmpty).length,
    naming: [state.naming.watch, state.naming.pencil].filter(nonEmpty).length,
    repetition: nonEmpty(state.repetition) ? 1 : 0,
    command: [state.command.tookPaper, state.command.foldedPaper, state.command.placedFloor].filter(
      (mark) => mark !== null
    ).length,
    reading: state.reading.correct !== null ? 1 : 0,
    writing: nonEmpty(state.writing) ? 1 : 0,
    copying: state.copying !== null ? 1 : 0,
  };

  const maxes: Record<SectionId, number> = {
    orientationTime: 5,
    orientationPlace: 5,
    registration: 3,
    attention: 5,
    delayedRecall: 3,
    naming: 2,
    repetition: 1,
    command: 3,
    reading: 1,
    writing: 1,
    copying: 1,
  };

  const result = {} as Record<SectionId, SectionResponseCount>;
  (Object.keys(maxes) as SectionId[]).forEach((id) => {
    result[id] = { done: counts[id], max: maxes[id] };
  });
  return result;
}

/** Phase 1 completeness: every required response/mark for the section is present. */
export function isSectionResponseComplete(id: SectionId, state: MMSEState): boolean {
  const count = sectionResponseCounts(state)[id];
  return count.done >= count.max;
}
