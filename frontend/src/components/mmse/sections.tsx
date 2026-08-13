import React, { useRef } from 'react';
import type { ItemState, MMSEState } from '../../mmse/state';
import { computeScores } from '../../mmse/state';
import {
  AI_CONFIDENCE_REVIEW_THRESHOLD,
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
import { evaluateMmseItem, extractApiError } from '../../api';
import {
  AIAssessmentPanel,
  AIScoredResponse,
  ExaminerInstructions,
  ExaminerScoring,
  PatientResponse,
  SectionShell,
  inputClass,
  useSpeechRecognition,
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
          patient&apos;s actual response — the AI service scores it automatically.
          Review the result and use Override only when needed. This is
          AI-assisted cognitive assessment, not a diagnosis.
        </ExaminerInstructions>
      }
    >
      {ORIENTATION_TIME_ITEMS.map((item) => (
        <AIScoredResponse
          key={item.key}
          section="orientation_time"
          itemKey={item.key}
          question={item.prompt}
          expected=""
          item={state.orientationTime.items[item.key]}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.orientationTime.items[item.key], patch);
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
          assessment location (set in{' '}
          <code className="text-gray-400">src/mmse/config.ts</code>). The AI scores
          each response; items without a configured expected answer are scored
          manually. Expected answers stay hidden from the patient.
        </ExaminerInstructions>
      }
    >
      {PLACE_ITEMS.map((item) => {
        const placeItem = state.orientationPlace.items[item.key];
        return (
          <AIScoredResponse
            key={item.key}
            section="orientation_place"
            itemKey={item.key}
            question={item.prompt}
            hint={
              item.expected
                ? `Expected answer: ${item.expected}`
                : 'Expected answer not configured for this location'
            }
            expected={item.expected}
            aiEnabled={Boolean(item.expected)}
            item={placeItem}
            update={(patch) =>
              update((draft) => {
                Object.assign(draft.orientationPlace.items[item.key], patch);
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
          Record the patient&apos;s repetition of each object — the AI scores it
          automatically. The same objects are used again in Delayed Recall.
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
        <AIScoredResponse
          key={obj}
          section="registration"
          itemKey={String(index + 1)}
          question={`Object ${index + 1}`}
          expected={obj}
          item={state.registration.items[index]}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.registration.items[index], patch);
            })
          }
        />
      ))}
    </SectionShell>
  );
};

const SpellWorldBlock: React.FC<SectionProps> = ({ state, update }) => {
  const busyRef = useRef(false);
  const fullResponse = state.attention.spellWorld.response;

  const patchLetter = (index: number, patch: Partial<ItemState>) =>
    update((draft) => {
      Object.assign(draft.attention.spellWorld.letters[index], patch);
    });

  const updateFullResponse = (value: string) => {
    if (value === fullResponse) return;
    update((draft) => {
      draft.attention.spellWorld.response = value;
      for (const letter of draft.attention.spellWorld.letters) {
        letter.response = '';
        letter.status = 'idle';
        letter.aiScore = null;
        letter.reviewRequired = false;
        letter.reviewed = false;
        letter.manual = null;
        letter.error = null;
      }
    });
  };

  const runLetterEvaluation = async (index: number, text: string) => {
    patchLetter(index, { status: 'assessing', error: null });
    try {
      const result = await evaluateMmseItem({
        section: 'attention_spell_world',
        item_key: String(index + 1),
        question: `Letter ${index + 1} of WORLD backwards`,
        response: text,
        expected: SPELL_WORLD_EXPECTED[index],
      });
      patchLetter(index, {
        status: 'assessed',
        aiScore: {
          correct: result.correct,
          confidence: result.confidence,
          reason: result.reason,
        },
        reviewRequired: result.confidence < AI_CONFIDENCE_REVIEW_THRESHOLD,
        reviewed: false,
        error: null,
      });
    } catch (err) {
      patchLetter(index, { status: 'error', error: extractApiError(err) });
    }
  };

  const evaluateFullResponse = async (text: string) => {
    if (busyRef.current) return;
    const chars = text
      .trim()
      .toUpperCase()
      .replace(/[^A-Z0-9]/g, '')
      .split('')
      .slice(0, 5);
    update((draft) => {
      for (let i = 0; i < 5; i += 1) {
        draft.attention.spellWorld.letters[i].response = chars[i] ?? '';
      }
    });
    busyRef.current = true;
    for (let i = 0; i < 5; i += 1) {
      await runLetterEvaluation(i, chars[i] ?? '');
    }
    busyRef.current = false;
  };

  const speechHook = useSpeechRecognition((transcript) => {
    updateFullResponse(transcript);
    window.setTimeout(() => {
      void evaluateFullResponse(transcript);
    }, 0);
  });
  const { supported: speechSupported, listening, error: speechError, start, stop } = speechHook;

  const micButton = (
    <button
      type="button"
      onClick={() => (listening ? stop() : start())}
      className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all duration-200 ${
        listening
          ? 'bg-rose-500/20 border-rose-400/40 text-rose-300'
          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
      }`}
    >
      {listening ? 'Stop' : 'Mic'}
    </button>
  );

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
      <div>
        <p className="text-sm font-medium text-white">Spell the word WORLD backwards</p>
        <p className="text-xs text-gray-500 mt-1">
          Record the full response; each letter is scored in its position.
        </p>
      </div>
      <div>
        <PatientResponse
          value={fullResponse}
          onChange={updateFullResponse}
          placeholder="Patient's full response"
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              if (fullResponse.trim() && !busyRef.current) {
                void evaluateFullResponse(fullResponse);
              }
            }
          }}
          rightSlot={speechSupported ? micButton : undefined}
        />
        {!speechSupported && (
          <p className="text-[11px] text-gray-500 mt-1.5">
            Speech input not supported in this browser — type the response instead.
          </p>
        )}
        {listening && (
          <p className="text-[11px] text-blue-300/80 mt-1.5">Listening… ask the patient to speak now.</p>
        )}
        {speechError && <p className="text-[11px] text-rose-300/80 mt-1.5">{speechError}</p>}
      </div>
      <div className="pt-4 border-t border-white/10 space-y-3">
        {SPELL_WORLD_EXPECTED.map((letter, index) => (
          <div key={index} className="rounded-xl border border-white/10 bg-white/[0.03] p-3 md:p-4">
            <div className="mb-2">
              <p className="text-sm font-medium text-white">Letter {index + 1}</p>
              <p className="text-xs text-gray-500">Expected: {letter}</p>
            </div>
            <AIAssessmentPanel
              item={state.attention.spellWorld.letters[index]}
              update={(patch) => patchLetter(index, patch)}
              onAssess={() =>
                void runLetterEvaluation(
                  index,
                  state.attention.spellWorld.letters[index].response
                )
              }
              idleEmptyHint="No letter provided for this position."
            />
          </div>
        ))}
      </div>
    </div>
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
          responses — the AI scores each step/letter automatically. Expected
          answers stay hidden from the patient.
          {isSerial7 ? (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Now I would like you to subtract 7 from 100 and keep
              subtracting 7 until I tell you to stop.&rdquo; Expected responses:{' '}
              {SERIAL_7_EXPECTED.join(', ')}.
            </p>
          ) : (
            <p className="mt-2 text-gray-400">
              Say: &ldquo;Spell the word WORLD backwards.&rdquo; Record the full
              response; each letter is scored in its position. Expected letters:{' '}
              {SPELL_WORLD_EXPECTED.join(', ')}.
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
            <AIScoredResponse
              key={`serial7-${index}`}
              section="attention_serial7"
              itemKey={String(index + 1)}
              question={`Response ${index + 1}`}
              hint={
                SERIAL_7_EXPECTED[index]
                  ? `Expected: ${SERIAL_7_EXPECTED[index]}`
                  : undefined
              }
              expected={SERIAL_7_EXPECTED[index] ?? ''}
              item={item}
              update={(patch) =>
                update((draft) => {
                  Object.assign(draft.attention.serial7[index], patch);
                })
              }
            />
          ))
        ) : (
          <SpellWorldBlock state={state} update={update} />
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
          me what those were?&rdquo; Record the patient&apos;s recall — the AI scores
          each object automatically. Objects presented during Registration:
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
        <AIScoredResponse
          key={index}
          section="delayed_recall"
          itemKey={String(index + 1)}
          question={`Response ${index + 1}`}
          expected={REGISTRATION_OBJECTS[index]}
          item={item}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.delayedRecall.items[index], patch);
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
          Show each item to the patient and ask: &ldquo;What is this?&rdquo; Record the
          patient&apos;s response — the AI scores it automatically and accepts common
          synonyms. The patient answers verbally.
        </ExaminerInstructions>
      }
    >
      {NAMING_ITEMS.map((item) => {
        const isWatch = item === 'Wristwatch';
        const naming = isWatch ? state.naming.watch : state.naming.pencil;
        const expected = isWatch ? 'wristwatch' : 'pencil';
        return (
          <AIScoredResponse
            key={item}
            section="naming"
            itemKey={isWatch ? 'wristwatch' : 'pencil'}
            question="What is this?"
            hint={`Present the object to the patient: ${item}`}
            expected={expected}
            item={naming}
            update={(patch) =>
              update((draft) => {
                if (isWatch) {
                  Object.assign(draft.naming.watch, patch);
                } else {
                  Object.assign(draft.naming.pencil, patch);
                }
              })
            }
          />
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
          ands, or buts.&rdquo; Record the patient&apos;s actual response — the AI scores
          it automatically, allowing harmless case, punctuation, or minor wording
          differences. Low-confidence results are flagged for review.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-6 text-center">
        <p className="text-sm text-gray-400 mb-1">Phrase to repeat</p>
        <p className="text-lg italic text-white">&ldquo;{REPETITION_PHRASE}&rdquo;</p>
      </div>
      <AIScoredResponse
        section="repetition"
        itemKey="phrase"
        question="Repeat the phrase for the patient, then record their response"
        expected={REPETITION_PHRASE}
        item={state.repetition}
        update={(patch) =>
          update((draft) => {
            Object.assign(draft.repetition, patch);
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
          your right hand, fold it in half, and put it on the floor.&rdquo; This is a
          physical/visual observation task, so it is scored by the examiner — not
          by the AI. Score each action independently.
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
          Automated vision scoring is not implemented yet, so the examiner records
          this observation manually.
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
          Ask: &ldquo;{WRITING_PROMPT}&rdquo; The only scoring criterion is that the
          sentence contains a noun and a verb. The AI checks only that criterion —
          never spelling, grammar, handwriting, or intelligence. The examiner can
          override the result.
        </ExaminerInstructions>
      }
    >
      <AIScoredResponse
        section="writing"
        itemKey="sentence"
        question={WRITING_PROMPT}
        expected=""
        item={state.writing}
        update={(patch) =>
          update((draft) => {
            Object.assign(draft.writing, patch);
          })
        }
        speech={false}
        multiline
        placeholder="Sentence written by the patient"
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
          exactly as possible. Automated vision scoring is not implemented yet —
          the examiner judges the copy manually.
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
