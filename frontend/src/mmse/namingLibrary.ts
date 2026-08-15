/**
 * Q6 Naming — controlled object library.
 *
 * The standardized MMSE Naming section stays exactly 2 points / 2 slots. This
 * library is a PRESENTATION / CONFIG layer: two objects are selected per
 * assessment session and fill the two existing Naming slots. The patient sees
 * the selected object image and is asked "What is this?". The expected answer is
 * examiner-only context (never shown to the patient).
 *
 * The two batch keys remain `naming.wristwatch` / `naming.pencil` (the two slots)
 * so the backend deterministic scorer keeps working unchanged; the `expected`
 * value sent for each slot is the selected object's canonical answer.
 */

export interface NamingObject {
  id: string;
  /** Examiner-facing display name (never shown to the patient). */
  label: string;
  /** Canonical expected answer used for scoring (e.g. "wristwatch"). */
  expected: string;
  /** Local asset path (controlled asset, no internet images). */
  image: string;
}

export const NAMING_OBJECTS: NamingObject[] = [
  { id: 'wristwatch', label: 'Wristwatch', expected: 'wristwatch', image: '/objects/wristwatch.svg' },
  { id: 'pencil', label: 'Pencil', expected: 'pencil', image: '/objects/pencil.svg' },
  { id: 'key', label: 'Key', expected: 'key', image: '/objects/key.svg' },
  { id: 'cup', label: 'Cup', expected: 'cup', image: '/objects/cup.svg' },
  { id: 'ball', label: 'Ball', expected: 'ball', image: '/objects/ball.svg' },
  { id: 'book', label: 'Book', expected: 'book', image: '/objects/book.svg' },
  { id: 'scissors', label: 'Scissors', expected: 'scissors', image: '/objects/scissors.svg' },
  { id: 'comb', label: 'Comb', expected: 'comb', image: '/objects/comb.svg' },
  { id: 'fork', label: 'Fork', expected: 'fork', image: '/objects/fork.svg' },
  { id: 'chair', label: 'Chair', expected: 'chair', image: '/objects/chair.svg' },
  { id: 'apple', label: 'Apple', expected: 'apple', image: '/objects/apple.svg' },
  { id: 'umbrella', label: 'Umbrella', expected: 'umbrella', image: '/objects/umbrella.svg' },
];

/** Default slot objects (the two canonical MMSE naming items). */
export const DEFAULT_NAMING_SELECTION: [string, string] = ['wristwatch', 'pencil'];

export function getNamingObject(id: string): NamingObject {
  return NAMING_OBJECTS.find((obj) => obj.id === id) ?? NAMING_OBJECTS[0];
}

/** Pick two distinct objects at random for the current assessment session. */
export function pickNamingObjects(): [string, string] {
  const shuffled = [...NAMING_OBJECTS].sort(() => Math.random() - 0.5);
  return [shuffled[0].id, shuffled[1].id];
}
