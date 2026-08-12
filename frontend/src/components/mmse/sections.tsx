import React from 'react';
import type { MMSEState } from '../../mmse/state';
import { computeScores } from '../../mmse/state';
import {
  ATTENTION_TASKS,
  COPYING_REFERENCE_IMAGE,
  NAMING_ITEMS,
  ORIENTATION_TIME_ITEMS,
  PLACE_ITEMS,
  READING_INSTRUCTION,
  REGISTRATION_OBJECTS,
  REPETITION_PHRASE,
  SERIAL_7_EXPECTED,
  SPELL_WORLD_EXPECTED,
  THREE_STEP_COMMAND_TEXT,
  WRITING_PROMPT,
} from '../../mmse/config';
import {
  ExaminerInstructions,
  ExaminerToggle,
  SectionShell,
} from './primitives';
import { DrawingCanvas } from './DrawingCanvas';

export interface SectionProps {
  state: MMSEState;
  update: (updater: (draft: MMSEState) => void) => void;
}

const inputClass =
  'w-full px-4 py-2.5 text-sm text-white bg-white/5 border border-white/10 rounded-lg focus:ring-2 focus:ring-blue-400/30 focus:border-blue-400/50 focus:bg-white/10 transition-all duration-200 placeholder:text-gray-600';

export const OrientationTimeSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Orientation to Time"
      score={computeScores(state).orientationTime}
      maxScore={5}
      instructions={
        <ExaminerInstructions>
          Ask the patient the five questions below one at a time. Give 1 point
          for each correct answer.
        </ExaminerInstructions>
      }
    >
      {ORIENTATION_TIME_ITEMS.map((item) => (
        <ExaminerToggle
          key={item.key}
          label={item.prompt}
          value={state.orientationTime[item.key]}
          onChange={(value) =>
            update((draft) => {
              draft.orientationTime[item.key] = value;
            })
          }
        />
      ))}
    </SectionShell>
  );
};

export const OrientationPlaceSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Orientation to Place"
      score={computeScores(state).orientationPlace}
      maxScore={5}
      instructions={
        <ExaminerInstructions>
          Ask the patient the five questions below. Correct answers depend on the
          assessment location. Record the patient&apos;s answer (optional) and
          mark each item correct or incorrect.
        </ExaminerInstructions>
      }
    >
      {PLACE_ITEMS.map((item) => {
        const placeItem = state.orientationPlace.items[item.key];
        return (
          <div key={item.key} className="space-y-2">
            <p className="text-sm font-medium text-white">{item.prompt}</p>
            {item.expected ? (
              <p className="text-xs text-gray-500">Expected answer: {item.expected}</p>
            ) : (
              <p className="text-xs text-gray-500">
                Expected answer: set per location in <code className="text-gray-400">src/mmse/config.ts</code>
              </p>
            )}
            <input
              type="text"
              value={placeItem.response}
              onChange={(event) =>
                update((draft) => {
                  draft.orientationPlace.items[item.key].response =
                    event.target.value;
                })
              }
              placeholder="Patient's answer (record)"
              className={inputClass}
            />
            <ExaminerToggle
              label="Answered correctly?"
              value={placeItem.correct}
              onChange={(value) =>
                update((draft) => {
                  draft.orientationPlace.items[item.key].correct =
                    value;
                })
              }
            />
          </div>
        );
      })}
    </SectionShell>
  );
};

export const RegistrationSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Registration"
      score={computeScores(state).registration}
      maxScore={3}
      instructions={
        <ExaminerInstructions>
          Say: &ldquo;I am going to name three objects. After I say them, I want you
          to repeat them back to me.&rdquo; Name the three objects one second apart.
          Give 1 point for each object the patient repeats correctly. The same
          objects are used again in the Delayed Recall section.
          <ul className="mt-3 space-y-1">
            {REGISTRATION_OBJECTS.map((obj) => (
              <li key={obj} className="font-medium text-white">
                {obj}
              </li>
            ))}
          </ul>
        </ExaminerInstructions>
      }
    >
      {REGISTRATION_OBJECTS.map((obj, index) => (
        <ExaminerToggle
          key={obj}
          label={`Object ${index + 1} — ${obj}`}
          value={state.registration.recalled[index]}
          onChange={(value) =>
            update((draft) => {
              draft.registration.recalled[index] = value;
            })
          }
        />
      ))}
    </SectionShell>
  );
};

export const AttentionSection: React.FC<SectionProps> = ({ state, update }) => {
  const isSerial7 = state.attention.task === 'serial7';
  const marks = isSerial7 ? state.attention.serial7 : state.attention.spellWorld;
  const expected = isSerial7 ? SERIAL_7_EXPECTED : SPELL_WORLD_EXPECTED;
  const itemLabel = isSerial7 ? 'Subtraction' : 'Letter';

  return (
    <SectionShell
      title="Attention & Calculation"
      score={computeScores(state).attention}
      maxScore={5}
      instructions={
        <ExaminerInstructions>
          Administer one of the two tasks below. Give 1 point for each correct
          response. Do not show the expected answers to the patient.
          {isSerial7 ? (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Now I would like you to subtract 7 from 100 and keep
              subtracting 7 until I tell you to stop.&rdquo; Expected responses:{' '}
              {SERIAL_7_EXPECTED.join(', ')}.
            </p>
          ) : (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Spell the word WORLD backwards.&rdquo; Expected responses:{' '}
              {SPELL_WORLD_EXPECTED.join(', ')} — 1 point per correct letter in the
              correct position.
            </p>
          )}
        </ExaminerInstructions>
      }
    >
      <div className="flex flex-wrap gap-2">
        {ATTENTION_TASKS.map((task) => (
          <button
            key={task.id}
            type="button"
            onClick={() =>
              update((draft) => {
                draft.attention.task = task.id;
              })
            }
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition-all duration-200 ${
              state.attention.task === task.id
                ? 'bg-blue-500/20 border-blue-400/40 text-blue-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            {task.label}
          </button>
        ))}
      </div>
      <div className="pt-1">
        {marks.map((mark, index) => (
          <ExaminerToggle
            key={`${state.attention.task}-${index}`}
            label={`${itemLabel} ${index + 1}`}
            hint={expected[index] ? `Expected: ${expected[index]}` : undefined}
            value={mark}
            onChange={(value) =>
              update((draft) => {
                if (isSerial7) {
                  draft.attention.serial7[index] = value;
                } else {
                  draft.attention.spellWorld[index] = value;
                }
              })
            }
          />
        ))}
      </div>
    </SectionShell>
  );
};

export const DelayedRecallSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Delayed Recall"
      score={computeScores(state).delayedRecall}
      maxScore={3}
      instructions={
        <ExaminerInstructions>
          Ask: &ldquo;Earlier I named three objects. What were they?&rdquo; Give 1
          point for each object correctly recalled. Objects presented during
          Registration:
          <ul className="mt-3 space-y-1">
            {REGISTRATION_OBJECTS.map((obj) => (
              <li key={obj} className="font-medium text-white">
                {obj}
              </li>
            ))}
          </ul>
        </ExaminerInstructions>
      }
    >
      {state.delayedRecall.recalled.map((mark, index) => (
        <ExaminerToggle
          key={index}
          label={`Object ${index + 1}`}
          value={mark}
          onChange={(value) =>
            update((draft) => {
              draft.delayedRecall.recalled[index] = value;
            })
          }
        />
      ))}
    </SectionShell>
  );
};

export const NamingSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Naming"
      score={computeScores(state).naming}
      maxScore={2}
      instructions={
        <ExaminerInstructions>
          Point to or show each item and ask: &ldquo;What is this?&rdquo; Give 1
          point for each item correctly named. The patient answers verbally.
        </ExaminerInstructions>
      }
    >
      {NAMING_ITEMS.map((item) => (
        <div
          key={item}
          className="rounded-xl border border-white/10 bg-white/[0.03] p-5 flex flex-col items-center gap-4"
        >
          <span className="text-lg font-semibold text-white">{item}</span>
          <ExaminerToggle
            label={item === 'Wristwatch' ? 'Watch named correctly' : 'Pencil named correctly'}
            value={item === 'Wristwatch' ? state.naming.watch : state.naming.pencil}
            onChange={(value) =>
              update((draft) => {
                if (item === 'Wristwatch') {
                  draft.naming.watch = value;
                } else {
                  draft.naming.pencil = value;
                }
              })
            }
          />
        </div>
      ))}
    </SectionShell>
  );
};

export const RepetitionSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Repetition"
      score={computeScores(state).repetition}
      maxScore={1}
      instructions={
        <ExaminerInstructions>
          Say: &ldquo;I am going to say a phrase. Please repeat it exactly: No ifs,
          ands, or buts.&rdquo; Give 1 point if the patient repeats it correctly on
          the first try.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-6 text-center">
        <p className="text-lg italic text-white">&ldquo;{REPETITION_PHRASE}&rdquo;</p>
      </div>
      <ExaminerToggle
        label="Repeated the phrase correctly"
        value={state.repetition}
        onChange={(value) =>
          update((draft) => {
            draft.repetition = value;
          })
        }
      />
    </SectionShell>
  );
};

export const ThreeStepCommandSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Three-Step Command"
      score={computeScores(state).command}
      maxScore={3}
      instructions={
        <ExaminerInstructions>
          Give the patient a blank piece of paper, then say: &ldquo;Take the paper in
          your right hand, fold it in half, and put it on the floor.&rdquo; Score each
          action independently.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4">
        <p className="text-sm text-gray-300 italic">
          &ldquo;{THREE_STEP_COMMAND_TEXT}&rdquo;
        </p>
      </div>
      <ExaminerToggle
        label="Took the paper in the right hand"
        value={state.command.tookPaper}
        onChange={(value) =>
          update((draft) => {
            draft.command.tookPaper = value;
          })
        }
      />
      <ExaminerToggle
        label="Folded the paper in half"
        value={state.command.foldedPaper}
        onChange={(value) =>
          update((draft) => {
            draft.command.foldedPaper = value;
          })
        }
      />
      <ExaminerToggle
        label="Put the paper on the floor"
        value={state.command.placedFloor}
        onChange={(value) =>
          update((draft) => {
            draft.command.placedFloor = value;
          })
        }
      />
    </SectionShell>
  );
};

export const ReadingSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Reading"
      score={computeScores(state).reading}
      maxScore={1}
      instructions={
        <ExaminerInstructions>
          Show the instruction card below to the patient. Do NOT read it aloud.
          Give 1 point if the patient reads it and performs the instruction.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-8 text-center">
        <p className="text-2xl font-semibold tracking-widest text-white">
          {READING_INSTRUCTION}
        </p>
      </div>
      <ExaminerToggle
        label="Followed the instruction (closed eyes)"
        value={state.reading}
        onChange={(value) =>
          update((draft) => {
            draft.reading = value;
          })
        }
      />
    </SectionShell>
  );
};

export const WritingSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Writing"
      score={computeScores(state).writing}
      maxScore={1}
      instructions={
        <ExaminerInstructions>
          Ask: &ldquo;{WRITING_PROMPT}&rdquo; Give 1 point if the sentence contains a
          noun and a verb. Do not grade spelling or grammar automatically.
        </ExaminerInstructions>
      }
    >
      <textarea
        value={state.writing.response}
        onChange={(event) =>
          update((draft) => {
            draft.writing.response = event.target.value;
          })
        }
        placeholder="Sentence recorded by the examiner"
        rows={4}
        className={`${inputClass} resize-none`}
      />
      <ExaminerToggle
        label="Sentence contains a noun and a verb"
        value={state.writing.correct}
        onChange={(value) =>
          update((draft) => {
            draft.writing.correct = value;
          })
        }
      />
    </SectionShell>
  );
};

export const CopyingSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Copying"
      score={computeScores(state).copying}
      maxScore={1}
      instructions={
        <ExaminerInstructions>
          Show the reference figure to the patient and ask them to copy it as
          exactly as possible. The examiner remains the final scorer.
        </ExaminerInstructions>
      }
    >
      {COPYING_REFERENCE_IMAGE ? (
        <img
          src={COPYING_REFERENCE_IMAGE}
          alt="Reference figure from the MMSE questionnaire"
          className="w-full max-h-64 object-contain rounded-xl border border-white/10 bg-white/[0.03]"
        />
      ) : (
        <div className="rounded-xl border border-dashed border-white/15 bg-white/[0.02] p-6 text-center">
          <p className="text-sm text-gray-400">
            The reference figure from the MMSE questionnaire is not bundled with
            this project.
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Supply the exact figure as an image asset and set its path in{' '}
            <code className="text-gray-300">src/mmse/config.ts</code>
            (COPYING_REFERENCE_IMAGE).
          </p>
        </div>
      )}
      <DrawingCanvas />
      <ExaminerToggle
        label="Copy is acceptable"
        value={state.copying}
        onChange={(value) =>
          update((draft) => {
            draft.copying = value;
          })
        }
      />
    </SectionShell>
  );
};
