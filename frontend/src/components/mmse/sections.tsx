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
  ExaminerScoring,
  PatientResponse,
  ScoredResponse,
  SectionShell,
  inputClass,
} from './primitives';
import { DrawingCanvas } from './DrawingCanvas';

export interface SectionProps {
  state: MMSEState;
  update: (updater: (draft: MMSEState) => void) => void;
}

export const OrientationTimeSection: React.FC<SectionProps> = ({ state, update }) => {
  return (
    <SectionShell
      title="Orientation to Time"
      score={computeScores(state).orientationTime}
      maxScore={5}
      instructions={
        <ExaminerInstructions>
          Ask the patient the five questions below one at a time. Record the
          patient&apos;s actual response, then give 1 point for each correct
          answer.
        </ExaminerInstructions>
      }
    >
      {ORIENTATION_TIME_ITEMS.map((item) => (
        <ScoredResponse
          key={item.key}
          prompt={item.prompt}
          response={state.orientationTime.items[item.key].response}
          onResponseChange={(value) =>
            update((draft) => {
              draft.orientationTime.items[item.key].response = value;
            })
          }
          responsePlaceholder="Patient's response"
          score={state.orientationTime.items[item.key].correct}
          onScoreChange={(value) =>
            update((draft) => {
              draft.orientationTime.items[item.key].correct = value;
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
          assessment location. Record the patient&apos;s actual response, then
          mark each item correct or incorrect. Do not show the expected answer to
          the patient.
        </ExaminerInstructions>
      }
    >
      {PLACE_ITEMS.map((item) => {
        const placeItem = state.orientationPlace.items[item.key];
        return (
          <ScoredResponse
            key={item.key}
            prompt={item.prompt}
            hint={
              item.expected
                ? `Expected answer: ${item.expected}`
                : 'Expected answer: set per location in src/mmse/config.ts'
            }
            response={placeItem.response}
            onResponseChange={(value) =>
              update((draft) => {
                draft.orientationPlace.items[item.key].response = value;
              })
            }
            responsePlaceholder="Patient's response"
            score={placeItem.correct}
            onScoreChange={(value) =>
              update((draft) => {
                draft.orientationPlace.items[item.key].correct = value;
              })
            }
          />
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
          Record the patient&apos;s repetition of each object, then give 1 point
          for each object repeated correctly. The same objects are used again in
          the Delayed Recall section.
          <ul className="mt-3 space-y-1">
            {REGISTRATION_OBJECTS.map((obj, index) => (
              <li key={obj} className="font-medium text-white">
                {index + 1}. {obj}
              </li>
            ))}
          </ul>
        </ExaminerInstructions>
      }
    >
      {REGISTRATION_OBJECTS.map((obj, index) => (
        <ScoredResponse
          key={obj}
          prompt={`Object ${index + 1}`}
          response={state.registration.items[index].response}
          onResponseChange={(value) =>
            update((draft) => {
              draft.registration.items[index].response = value;
            })
          }
          responsePlaceholder="Patient's response"
          score={state.registration.items[index].correct}
          onScoreChange={(value) =>
            update((draft) => {
              draft.registration.items[index].correct = value;
            })
          }
        />
      ))}
    </SectionShell>
  );
};

export const AttentionSection: React.FC<SectionProps> = ({ state, update }) => {
  const isSerial7 = state.attention.task === 'serial7';

  return (
    <SectionShell
      title="Attention & Calculation"
      score={computeScores(state).attention}
      maxScore={5}
      instructions={
        <ExaminerInstructions>
          Administer one of the two tasks below. Record the patient&apos;s
          responses, then give 1 point for each correct response. Do not show the
          expected answers to the patient.
          {isSerial7 ? (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Now I would like you to subtract 7 from 100 and keep
              subtracting 7 until I tell you to stop.&rdquo; Expected responses:{' '}
              {SERIAL_7_EXPECTED.join(', ')}.
            </p>
          ) : (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Spell the word WORLD backwards.&rdquo; Record the full
              response, then score each letter in the correct position. Expected
              letters: {SPELL_WORLD_EXPECTED.join(', ')}.
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
        {isSerial7 ? (
          state.attention.serial7.map((item, index) => (
            <ScoredResponse
              key={`serial7-${index}`}
              prompt={`Response ${index + 1}`}
              hint={
                SERIAL_7_EXPECTED[index]
                  ? `Expected: ${SERIAL_7_EXPECTED[index]}`
                  : undefined
              }
              response={item.response}
              onResponseChange={(value) =>
                update((draft) => {
                  draft.attention.serial7[index].response = value;
                })
              }
              responsePlaceholder="Patient's response"
              score={item.correct}
              onScoreChange={(value) =>
                update((draft) => {
                  draft.attention.serial7[index].correct = value;
                })
              }
            />
          ))
        ) : (
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
            <div>
              <p className="text-sm font-medium text-white">
                Spell the word WORLD backwards
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Score each letter only if it is in the correct position.
              </p>
            </div>
            <PatientResponse
              value={state.attention.spellWorld.response}
              onChange={(value) =>
                update((draft) => {
                  draft.attention.spellWorld.response = value;
                })
              }
              placeholder="Patient's full response"
            />
            <div className="pt-4 border-t border-white/10 space-y-3">
              {state.attention.spellWorld.letters.map((item, index) => (
                <ExaminerScoring
                  key={`letter-${index}`}
                  label={`Letter ${index + 1}`}
                  labelVariant="row"
                  hint={
                    SPELL_WORLD_EXPECTED[index]
                      ? `Expected: ${SPELL_WORLD_EXPECTED[index]}`
                      : undefined
                  }
                  value={item.correct}
                  onChange={(value) =>
                    update((draft) => {
                      draft.attention.spellWorld.letters[index].correct = value;
                    })
                  }
                />
              ))}
            </div>
          </div>
        )}
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
          Ask: &ldquo;Earlier I told you the names of three things. Can you tell
          me what those were?&rdquo; Record the patient&apos;s recall, then give 1
          point for each object correctly recalled. Objects presented during
          Registration:
          <ul className="mt-3 space-y-1">
            {REGISTRATION_OBJECTS.map((obj, index) => (
              <li key={obj} className="font-medium text-white">
                {index + 1}. {obj}
              </li>
            ))}
          </ul>
        </ExaminerInstructions>
      }
    >
      {state.delayedRecall.items.map((item, index) => (
        <ScoredResponse
          key={index}
          prompt={`Response ${index + 1}`}
          response={item.response}
          onResponseChange={(value) =>
            update((draft) => {
              draft.delayedRecall.items[index].response = value;
            })
          }
          responsePlaceholder="Patient's response"
          score={item.correct}
          onScoreChange={(value) =>
            update((draft) => {
              draft.delayedRecall.items[index].correct = value;
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
          Point to or show each item and ask: &ldquo;What is this?&rdquo; Record
          the patient&apos;s response, then give 1 point for each item correctly
          named. The patient answers verbally.
        </ExaminerInstructions>
      }
    >
      {NAMING_ITEMS.map((item) => {
        const isWatch = item === 'Wristwatch';
        const naming = isWatch ? state.naming.watch : state.naming.pencil;
        return (
          <div
            key={item}
            className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4"
          >
            <div className="flex flex-col items-center gap-1 text-center">
              <span className="text-lg font-semibold text-white">{item}</span>
              <p className="text-sm text-gray-400">&ldquo;What is this?&rdquo;</p>
            </div>
            <PatientResponse
              value={naming.response}
              onChange={(value) =>
                update((draft) => {
                  if (isWatch) {
                    draft.naming.watch.response = value;
                  } else {
                    draft.naming.pencil.response = value;
                  }
                })
              }
              placeholder="Patient's response"
            />
            <div className="pt-4 border-t border-white/10">
              <ExaminerScoring
                label={
                  isWatch
                    ? 'Examiner scoring — watch named correctly'
                    : 'Examiner scoring — pencil named correctly'
                }
                value={naming.correct}
                onChange={(value) =>
                  update((draft) => {
                    if (isWatch) {
                      draft.naming.watch.correct = value;
                    } else {
                      draft.naming.pencil.correct = value;
                    }
                  })
                }
              />
            </div>
          </div>
        );
      })}
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
          ands, or buts.&rdquo; Record the patient&apos;s actual response, then give
          1 point if it is repeated correctly on the first try.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-6 text-center">
        <p className="text-sm text-gray-400 mb-1">Phrase to repeat</p>
        <p className="text-lg italic text-white">&ldquo;{REPETITION_PHRASE}&rdquo;</p>
      </div>
      <ScoredResponse
        prompt="Repeat the phrase for the patient, then record their response"
        response={state.repetition.response}
        onResponseChange={(value) =>
          update((draft) => {
            draft.repetition.response = value;
          })
        }
        responsePlaceholder="Patient's response"
        score={state.repetition.correct}
        onScoreChange={(value) =>
          update((draft) => {
            draft.repetition.correct = value;
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
          action independently based on what you observe. The patient does not
          type anything.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4">
        <p className="text-sm text-gray-400 mb-1">Command to give the patient</p>
        <p className="text-sm text-gray-300 italic">
          &ldquo;{THREE_STEP_COMMAND_TEXT}&rdquo;
        </p>
      </div>
      <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-blue-300/80 pt-1">
        Examiner observations
      </p>
      <ExaminerScoring
        label="Took the paper in the right hand"
        labelVariant="row"
        correctLabel="Done correctly"
        incorrectLabel="Incorrect"
        value={state.command.tookPaper}
        onChange={(value) =>
          update((draft) => {
            draft.command.tookPaper = value;
          })
        }
      />
      <ExaminerScoring
        label="Folded the paper in half"
        labelVariant="row"
        correctLabel="Done correctly"
        incorrectLabel="Incorrect"
        value={state.command.foldedPaper}
        onChange={(value) =>
          update((draft) => {
            draft.command.foldedPaper = value;
          })
        }
      />
      <ExaminerScoring
        label="Put the paper on the floor"
        labelVariant="row"
        correctLabel="Done correctly"
        incorrectLabel="Incorrect"
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
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
        <div>
          <p className="text-sm font-medium text-white">
            Did the patient read and perform the instruction?
          </p>
        </div>
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-400 mb-2">
            Examiner note <span className="normal-case tracking-normal text-gray-500">(optional)</span>
          </p>
          <input
            type="text"
            value={state.reading.note}
            onChange={(event) =>
              update((draft) => {
                draft.reading.note = event.target.value;
              })
            }
            placeholder="Optional note"
            className={inputClass}
          />
        </div>
        <div className="pt-4 border-t border-white/10">
          <ExaminerScoring
            value={state.reading.correct}
            onChange={(value) =>
              update((draft) => {
                draft.reading.correct = value;
              })
            }
          />
        </div>
      </div>
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
          noun and a verb. Do not grade spelling or grammar automatically — the
          examiner decides.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
        <p className="text-sm font-medium text-white">{WRITING_PROMPT}</p>
        <PatientResponse
          multiline
          value={state.writing.response}
          onChange={(value) =>
            update((draft) => {
              draft.writing.response = value;
            })
          }
          placeholder="Sentence written by the patient"
        />
        <div className="pt-4 border-t border-white/10">
          <ExaminerScoring
            hint="Give 1 point if the sentence contains a noun and a verb"
            value={state.writing.correct}
            onChange={(value) =>
              update((draft) => {
                draft.writing.correct = value;
              })
            }
          />
        </div>
      </div>
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
          exactly as possible. The examiner remains the final scorer — the drawing
          is never auto-scored.
        </ExaminerInstructions>
      }
    >
      <div className="space-y-3">
        <p className="text-sm font-medium text-white">Please copy this picture.</p>
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
        <div className="pt-3 border-t border-white/10">
          <ExaminerScoring
            hint="Judge the copy visually; do not auto-score the drawing"
            value={state.copying}
            onChange={(value) =>
              update((draft) => {
                draft.copying = value;
              })
            }
          />
        </div>
      </div>
    </SectionShell>
  );
};
