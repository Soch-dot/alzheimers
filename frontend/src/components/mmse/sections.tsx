import React from 'react';
import type { ItemState, MMSEState, MmsePhase } from '../../mmse/state';
import { computeScores } from '../../mmse/state';
import { sectionResponseCounts } from '../../mmse/batch';
import {
  ATTENTION_TASKS,
  COPYING_REFERENCE_IMAGE,
  LOCATION_FIELDS,
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
  AIResultPanel,
  AIScoredResponse,
  ExaminerInstructions,
  ExaminerScoring,
  PatientResponse,
  SectionShell,
  inputClass,
  useSpeechRecognition,
} from './primitives';
import { Q11PhotoAssessment } from './Q11PhotoAssessment';

export interface SectionProps {
  state: MMSEState;
  update: (updater: (draft: MMSEState) => void) => void;
  phase: MmsePhase;
  /** Re-run the batch assessment for still-unscored items. */
  onRetry: () => void;
}

export const OrientationTimeSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Orientation to Time"
      score={computeScores(state).orientationTime}
      maxScore={5}
      phase={phase}
      responseCount={counts.orientationTime.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Ask the patient the five questions below one at a time and record their
          actual responses. Responses are collected first; when all sections are
          complete, one &ldquo;Assess MMSE with AI&rdquo; action scores everything in a
          single batch. This is AI-assisted cognitive assessment, not a diagnosis.
        </ExaminerInstructions>
      }
    >
      {ORIENTATION_TIME_ITEMS.map((item) => (
        <AIScoredResponse
          key={item.key}
          question={item.prompt}
          item={state.orientationTime.items[item.key]}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.orientationTime.items[item.key], patch);
            })
          }
          phase={phase}
          onRetry={onRetry}
        />
      ))}
    </SectionShell>
  );
};

export const OrientationPlaceSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  const locationIncomplete = LOCATION_FIELDS.some(
    (field) => state.location[field.key].trim() === ''
  );

  return (
    <SectionShell
      title="Orientation to Place"
      score={computeScores(state).orientationPlace}
      maxScore={5}
      phase={phase}
      responseCount={counts.orientationPlace.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Set the assessment location, then ask the patient the five questions
          below one at a time and record their actual responses. The reference
          answers are set by the examiner below and are used by the AI during the
          batch assessment — they are never shown to the patient.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 md:p-5 space-y-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-blue-300/80">
            Assessment location{' '}
            <span className="normal-case tracking-normal text-gray-500">
              — for the examiner, not the patient
            </span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Enter where this assessment is taking place. These values are the
            reference answers the AI uses to evaluate the patient&apos;s responses.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {LOCATION_FIELDS.map((field) => (
            <div key={field.key}>
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-400 mb-1">
                {field.label}
              </p>
              <input
                type="text"
                value={state.location[field.key]}
                onChange={(event) =>
                  update((draft) => {
                    draft.location[field.key] = event.target.value;
                  })
                }
                placeholder={field.placeholder}
                className={inputClass}
              />
            </div>
          ))}
        </div>
        {locationIncomplete && (
          <p className="text-xs text-amber-300/80">
            Finish setting the assessment location — all five fields are required
            before the assessment can continue.
          </p>
        )}
      </div>

      {PLACE_ITEMS.map((item) => {
        const placeItem = state.orientationPlace.items[item.key];
        const configured = state.location[item.key].trim() !== '';
        return (
          <AIScoredResponse
            key={item.key}
            question={item.prompt}
            disabledNotice={
              configured
                ? undefined
                : 'Set this field in the assessment location above to enable AI assessment.'
            }
            item={placeItem}
            update={(patch) =>
              update((draft) => {
                Object.assign(draft.orientationPlace.items[item.key], patch);
              })
            }
            phase={phase}
            onRetry={onRetry}
          />
        );
      })}
    </SectionShell>
  );
};

export const RegistrationSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Registration"
      score={computeScores(state).registration}
      maxScore={3}
      phase={phase}
      responseCount={counts.registration.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Say: &ldquo;I am going to name three objects. After I say them, I want you
          to repeat them back to me.&rdquo; Name the three objects one second apart.
          Record the patient&apos;s repetition of each object. The same objects are
          used again in Delayed Recall.
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
          question={`Object ${index + 1}`}
          item={state.registration.items[index]}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.registration.items[index], patch);
            })
          }
          phase={phase}
          onRetry={onRetry}
        />
      ))}
    </SectionShell>
  );
};

const SpellWorldBlock: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const fullResponse = state.attention.spellWorld.response;

  const patchLetter = (index: number, patch: Partial<ItemState>) =>
    update((draft) => {
      Object.assign(draft.attention.spellWorld.letters[index], patch);
    });

  const updateFullResponse = (value: string) => {
    if (value === fullResponse) return;
    const chars = value
      .trim()
      .toUpperCase()
      .replace(/[^A-Z0-9]/g, '')
      .split('')
      .slice(0, 5);
    update((draft) => {
      draft.attention.spellWorld.response = value;
      draft.attention.spellWorld.letters.forEach((letter, index) => {
        letter.response = chars[index] ?? '';
        letter.status = 'idle';
        letter.aiScore = null;
        letter.reviewRequired = false;
        letter.reviewed = false;
        letter.manual = null;
        letter.error = null;
      });
    });
  };

  const speechHook = useSpeechRecognition((transcript) => {
    updateFullResponse(transcript);
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
          Record the full response. Each letter is scored in its position during the
          batch assessment.
        </p>
      </div>
      <div>
        <PatientResponse
          value={fullResponse}
          onChange={updateFullResponse}
          placeholder="Patient's full response"
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
      {phase === 'collect' ? (
        fullResponse.trim() ? (
          <div className="pt-3 border-t border-white/10 flex items-center gap-2">
            <span className="text-emerald-300">✓</span>
            <p className="text-xs text-gray-400">Response recorded</p>
          </div>
        ) : null
      ) : (
        <div className="pt-4 border-t border-white/10 space-y-3">
          {SPELL_WORLD_EXPECTED.map((letter, index) => {
            const letterItem = state.attention.spellWorld.letters[index];
            return (
              <div key={index} className="rounded-xl border border-white/10 bg-white/[0.03] p-3 md:p-4">
                <div className="mb-2">
                  <p className="text-sm font-medium text-white">Letter {index + 1}</p>
                  <p className="text-xs text-gray-500">
                    Expected: {letter}
                    {letterItem.response ? ` · Recorded: ${letterItem.response}` : ''}
                  </p>
                </div>
                <AIResultPanel
                  item={letterItem}
                  update={(patch) => patchLetter(index, patch)}
                  onRetry={onRetry}
                  notAssessedHint="No AI result for this letter yet."
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const AttentionSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const isSerial7 = state.attention.task === 'serial7';
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Attention & Calculation"
      score={computeScores(state).attention}
      maxScore={5}
      phase={phase}
      responseCount={counts.attention.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Administer one of the two tasks below and record the patient&apos;s
          responses. Expected answers stay hidden from the patient; they are only
          used by the AI during the batch assessment.
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
              question={`Response ${index + 1}`}
              hint={
                SERIAL_7_EXPECTED[index]
                  ? `Expected: ${SERIAL_7_EXPECTED[index]}`
                  : undefined
              }
              item={item}
              update={(patch) =>
                update((draft) => {
                  Object.assign(draft.attention.serial7[index], patch);
                })
              }
              phase={phase}
              onRetry={onRetry}
            />
          ))
        ) : (
          <SpellWorldBlock state={state} update={update} phase={phase} onRetry={onRetry} />
        )}
      </div>
    </SectionShell>
  );
};

export const DelayedRecallSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Delayed Recall"
      score={computeScores(state).delayedRecall}
      maxScore={3}
      phase={phase}
      responseCount={counts.delayedRecall.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Ask: &ldquo;Earlier I told you the names of three things. Can you tell
          me what those were?&rdquo; Record the patient&apos;s recall of each object.
          Objects presented during Registration:
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
          question={`Response ${index + 1}`}
          item={item}
          update={(patch) =>
            update((draft) => {
              Object.assign(draft.delayedRecall.items[index], patch);
            })
          }
          phase={phase}
          onRetry={onRetry}
        />
      ))}
    </SectionShell>
  );
};

export const NamingSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Naming"
      score={computeScores(state).naming}
      maxScore={2}
      phase={phase}
      responseCount={counts.naming.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Show each item to the patient and ask: &ldquo;What is this?&rdquo; Record the
          patient&apos;s response. The AI accepts common synonyms. The patient answers
          verbally.
        </ExaminerInstructions>
      }
    >
      {NAMING_ITEMS.map((item) => {
        const isWatch = item === 'Wristwatch';
        const naming = isWatch ? state.naming.watch : state.naming.pencil;
        return (
          <AIScoredResponse
            key={item}
            question="What is this?"
            hint={`Present the object to the patient: ${item}`}
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
            phase={phase}
            onRetry={onRetry}
          />
        );
      })}
    </SectionShell>
  );
};

export const RepetitionSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Repetition"
      score={computeScores(state).repetition}
      maxScore={1}
      phase={phase}
      responseCount={counts.repetition.done}
      kind="ai"
      instructions={
        <ExaminerInstructions>
          Say: &ldquo;I am going to say a phrase. Please repeat it exactly: No ifs,
          ands, or buts.&rdquo; Record the patient&apos;s actual response. The AI allows
          harmless case, punctuation, or minor wording differences; low-confidence
          results are flagged for review.
        </ExaminerInstructions>
      }
    >
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-6 text-center">
        <p className="text-sm text-gray-400 mb-1">Phrase to repeat</p>
        <p className="text-lg italic text-white">&ldquo;{REPETITION_PHRASE}&rdquo;</p>
      </div>
      <AIScoredResponse
        question="Repeat the phrase for the patient, then record their response"
        item={state.repetition}
        update={(patch) =>
          update((draft) => {
            Object.assign(draft.repetition, patch);
          })
        }
        phase={phase}
        onRetry={onRetry}
      />
    </SectionShell>
  );
};

export const ThreeStepCommandSection: React.FC<SectionProps> = ({ state, update, phase }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Three-Step Command"
      score={computeScores(state).command}
      maxScore={3}
      phase={phase}
      responseCount={counts.command.done}
      kind="observation"
      instructions={
        <ExaminerInstructions>
          Give the patient a blank piece of paper, then say: &ldquo;Take the paper in
          your right hand, fold it in half, and put it on the floor.&rdquo; This is an
          observation-based assessment — record each action you observe. AI vision
          assistance for physical-task observation will be added in a later
          milestone.
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

export const ReadingSection: React.FC<SectionProps> = ({ state, update, phase }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Reading"
      score={computeScores(state).reading}
      maxScore={1}
      phase={phase}
      responseCount={counts.reading.done}
      kind="observation"
      instructions={
        <ExaminerInstructions>
          Show the instruction card below to the patient. Do NOT read it aloud.
          This is an observation-based assessment — record whether the patient
          reads and performs the instruction. AI vision assistance will be added
          in a later milestone.
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

export const WritingSection: React.FC<SectionProps> = ({ state, update, phase, onRetry }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Writing"
      score={computeScores(state).writing}
      maxScore={1}
      phase={phase}
      responseCount={counts.writing.done}
      kind="ai"
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
        question={WRITING_PROMPT}
        item={state.writing}
        update={(patch) =>
          update((draft) => {
            Object.assign(draft.writing, patch);
          })
        }
        speech={false}
        multiline
        placeholder="Sentence written by the patient"
        phase={phase}
        onRetry={onRetry}
      />
    </SectionShell>
  );
};

export const CopyingSection: React.FC<SectionProps> = ({ state, update, phase }) => {
  const counts = sectionResponseCounts(state);
  return (
    <SectionShell
      title="Copying"
      score={computeScores(state).copying}
      maxScore={1}
      phase={phase}
      responseCount={counts.copying.done}
      kind="observation"
      instructions={
        <ExaminerInstructions>
          Show the reference figure to the patient and ask them to copy it as
          exactly as possible on paper. Take a clear photo of the patient&rsquo;s
          drawing and analyze it. The AI checks the copy against the reference —
          the examiner can accept or override the result.
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
              this version.
            </p>
            <p className="text-sm text-gray-400 mt-2">
              It will be added in a later milestone.
            </p>
          </div>
        )}
        <Q11PhotoAssessment
          copying={state.copying}
          updateCopying={(patch) =>
            update((draft) => {
              Object.assign(draft.copying, patch);
            })
          }
        />
      </div>
    </SectionShell>
  );
};