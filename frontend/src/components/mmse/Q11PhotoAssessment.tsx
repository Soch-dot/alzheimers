import React, { useRef, useState } from 'react';
import { classifyCopyingError, evaluateCopyingImage } from '../../api';
import type { CopyingState } from '../../mmse/state';

interface Q11PhotoAssessmentProps {
  copying: CopyingState;
  updateCopying: (patch: Partial<CopyingState>) => void;
}

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

const ghostButtonClass =
  'px-4 py-1.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed';

/**
 * Question 11 photo flow.
 *
 * The patient copies the reference figure on paper; the examiner takes or
 * uploads a photo. The photo preview is an in-memory data URL that lives only
 * for the current assessment — it is never written to disk, localStorage, or
 * the backend. Analysis runs on the explicit "Analyze Drawing" action (no
 * auto-call) against the dedicated vision endpoint.
 */
export const Q11PhotoAssessment: React.FC<Q11PhotoAssessmentProps> = ({
  copying,
  updateCopying,
}) => {
  const captureRef = useRef<HTMLInputElement | null>(null);
  const uploadRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [showManual, setShowManual] = useState(false);

  const cameraSupported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia;

  const resetToEmpty = () => {
    setCameraError(null);
    setShowManual(false);
    updateCopying({
      status: 'empty',
      previewData: null,
      previewName: '',
      aiScore: null,
      reviewRequired: false,
      reviewed: false,
      manual: null,
      errorKind: null,
      errorDetail: null,
    });
  };

  const readFile = (file: File | null) => {
    setCameraError(null);
    if (!file) return;
    const supported =
      /^image\/(jpeg|png|webp)$/i.test(file.type) ||
      /\.(jpe?g|png|webp)$/i.test(file.name);
    if (!supported) {
      updateCopying({
        status: 'error',
        errorKind: 'upload',
        errorDetail: 'Unsupported image type. Use JPEG, PNG, or WebP.',
      });
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      updateCopying({
        status: 'error',
        errorKind: 'upload',
        errorDetail: 'The image exceeds the maximum allowed size (10 MB).',
      });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = typeof reader.result === 'string' ? reader.result : null;
      updateCopying({
        status: 'photo',
        previewData: dataUrl,
        previewName: file.name,
        aiScore: null,
        reviewRequired: false,
        reviewed: false,
        manual: null,
        errorKind: null,
        errorDetail: null,
      });
    };
    reader.onerror = () => {
      updateCopying({
        status: 'error',
        errorKind: 'upload',
        errorDetail: 'The photo could not be read. Please take or upload another photo.',
      });
    };
    reader.readAsDataURL(file);
  };

  const handleCapture = async () => {
    setCameraError(null);
    if (!cameraSupported) {
      setCameraError('Camera capture is not supported in this browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((track) => track.stop());
      captureRef.current?.click();
    } catch (err) {
      const name = (err as { name?: string })?.name;
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setCameraError('Camera access was not granted.');
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setCameraError('No camera was found. Use Upload Photo instead.');
      } else {
        setCameraError('The camera could not be started. Use Upload Photo instead.');
      }
    }
  };

  const analyze = async () => {
    if (!copying.previewData || busy) return;
    setBusy(true);
    setCameraError(null);
    setShowManual(false);
    updateCopying({ status: 'analyzing', errorKind: null, errorDetail: null });
    try {
      const res = await evaluateCopyingImage(copying.previewData);
      updateCopying({
        status: 'assessed',
        aiScore: {
          correct: res.correct,
          confidence: res.confidence,
          reason: res.reason,
        },
        reviewRequired: res.review_required,
        reviewed: false,
        manual: null,
      });
    } catch (err) {
      const { kind, detail } = classifyCopyingError(err);
      updateCopying({
        status: 'error',
        errorKind: kind,
        errorDetail: kind === 'upload' ? detail : null,
      });
    } finally {
      setBusy(false);
    }
  };

  const errorMessage = (): { title: string; actions: React.ReactNode } | null => {
    switch (copying.errorKind) {
      case 'blank':
        return {
          title: 'No drawing detected. Please submit a clear photo of the patient\'s drawing.',
          actions: (
            <>
              <button type="button" onClick={handleCapture} className={ghostButtonClass}>
                Retake
              </button>
              <button
                type="button"
                onClick={() => uploadRef.current?.click()}
                className={ghostButtonClass}
              >
                Choose Another
              </button>
            </>
          ),
        };
      case 'upload':
        return {
          title:
            copying.errorDetail ||
            'The photo could not be read. Please take or upload another photo.',
          actions: (
            <>
              <button type="button" onClick={handleCapture} className={ghostButtonClass}>
                Retake
              </button>
              <button
                type="button"
                onClick={() => uploadRef.current?.click()}
                className={ghostButtonClass}
              >
                Choose Another
              </button>
            </>
          ),
        };
      case 'timeout':
        return {
          title: 'Vision assessment timed out.',
          actions: (
            <>
              <button
                type="button"
                onClick={() => void analyze()}
                disabled={busy || !copying.previewData}
                className={ghostButtonClass}
              >
                Retry
              </button>
              <button type="button" onClick={handleCapture} className={ghostButtonClass}>
                Retake
              </button>
            </>
          ),
        };
      case 'invalid':
        return {
          title: 'Vision assessment returned an invalid result.',
          actions: (
            <button
              type="button"
              onClick={() => void analyze()}
              disabled={busy || !copying.previewData}
              className={ghostButtonClass}
            >
              Retry
            </button>
          ),
        };
      case 'unavailable':
        return {
          title: 'Vision assessment unavailable.',
          actions: (
            <button
              type="button"
              onClick={() => void analyze()}
              disabled={busy || !copying.previewData}
              className={ghostButtonClass}
            >
              Retry
            </button>
          ),
        };
      default:
        return null;
    }
  };

  const renderResult = () => {
    const ai = copying.aiScore;
    if (!ai) return null;
    const effective = copying.manual ?? ai.correct;
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`text-sm font-semibold ${ai.correct ? 'text-emerald-300' : 'text-rose-300'}`}
          >
            {ai.correct ? '✓ Correct response' : '✕ Incorrect response'}
          </span>
          {copying.reviewRequired && !copying.reviewed && (
            <span className="text-xs font-semibold text-amber-300">⚠ Review required</span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <p className="text-xs text-gray-400">
            AI confidence: {Math.round(ai.confidence * 100)}%
          </p>
          <p className="text-xs text-gray-500">
            Score: {effective === null ? '—' : effective ? '1' : '0'} / 1
          </p>
        </div>
        {ai.reason && (
          <p className="text-xs text-gray-500 italic">“{ai.reason}”</p>
        )}
        {copying.reviewRequired && !copying.reviewed ? (
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => updateCopying({ reviewed: true })}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold border border-emerald-400/30 bg-emerald-500/10 text-emerald-300 transition-all duration-200 hover:bg-emerald-500/20"
            >
              Accept AI result
            </button>
            <button type="button" onClick={() => setShowManual((v) => !v)} className={ghostButtonClass}>
              Override
            </button>
          </div>
        ) : (
          <div>
            {copying.manual !== null ? (
              <p className="text-xs text-gray-400">
                Examiner override:{' '}
                <span className={copying.manual ? 'text-emerald-300' : 'text-rose-300'}>
                  {copying.manual ? 'Correct' : 'Incorrect'}
                </span>
              </p>
            ) : (
              <button
                type="button"
                onClick={() => setShowManual((v) => !v)}
                className="text-xs text-gray-500 hover:text-white transition-colors"
              >
                {showManual ? 'Hide override' : 'Review / Override'}
              </button>
            )}
            {showManual && (
              <div className="flex items-center gap-2 mt-3">
                <span className="text-xs text-gray-500">Manual score:</span>
                <button
                  type="button"
                  onClick={() => updateCopying({ manual: true })}
                  className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
                    copying.manual === true
                      ? 'bg-emerald-500/20 border-emerald-400/40 text-emerald-300'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Correct
                </button>
                <button
                  type="button"
                  onClick={() => updateCopying({ manual: false })}
                  className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
                    copying.manual === false
                      ? 'bg-rose-500/20 border-rose-400/40 text-rose-300'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Incorrect
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const showPreview = copying.status !== 'empty';

  return (
    <div className="space-y-4">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-400 mb-2">
          Patient drawing
        </p>

        {!showPreview ? (
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => void handleCapture()}
                className="px-4 py-2 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10"
              >
                Take Photo
              </button>
              <button
                type="button"
                onClick={() => uploadRef.current?.click()}
                className="px-4 py-2 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10"
              >
                Upload Photo
              </button>
            </div>
            {cameraError ? (
              <p className="text-xs text-rose-300/80 mt-2">{cameraError}</p>
            ) : !cameraSupported ? (
              <p className="text-[11px] text-gray-500 mt-2">
                Camera capture is not supported in this browser — you can upload a photo instead.
              </p>
            ) : null}
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-400 mb-2">
                Photo preview
              </p>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden">
                {copying.previewData ? (
                  <img
                    src={copying.previewData}
                    alt="Patient's copy of the reference figure"
                    className="w-full max-h-80 object-contain bg-black/30"
                  />
                ) : (
                  <div className="h-40 flex items-center justify-center">
                    <p className="text-xs text-gray-600">No preview available</p>
                  </div>
                )}
              </div>
              {copying.previewName && (
                <p className="text-[11px] text-gray-600 mt-1.5 break-all">{copying.previewName}</p>
              )}
            </div>

            {copying.status === 'photo' && (
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => void handleCapture()} className={ghostButtonClass}>
                  Retake
                </button>
                <button
                  type="button"
                  onClick={() => uploadRef.current?.click()}
                  className={ghostButtonClass}
                >
                  Choose Another
                </button>
                <button
                  type="button"
                  onClick={() => void analyze()}
                  disabled={busy}
                  className="px-5 py-1.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white transition-all duration-200 hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Analyze Drawing
                </button>
              </div>
            )}

            {copying.status === 'analyzing' && (
              <div className="flex items-center gap-3 rounded-xl border border-blue-400/20 bg-blue-500/10 px-4 py-3">
                <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                <div>
                  <p className="text-sm text-blue-200">Analyzing drawing…</p>
                  <p className="text-xs text-blue-300/70 mt-0.5">
                    This may take up to a minute on the local vision model.
                  </p>
                </div>
              </div>
            )}

            {copying.status === 'assessed' && (
              <div className="space-y-3">
                {renderResult()}
                <button
                  type="button"
                  onClick={resetToEmpty}
                  className="text-xs text-gray-500 hover:text-white transition-colors"
                >
                  Take another photo
                </button>
              </div>
            )}

            {copying.status === 'error' && (() => {
              const info = errorMessage();
              if (!info) return null;
              return (
                <div className="space-y-3">
                  <div className="rounded-xl border border-rose-400/20 bg-rose-500/10 px-4 py-3">
                    <p className="text-sm text-rose-200">{info.title}</p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">{info.actions}</div>
                </div>
              );
            })()}
          </div>
        )}
      </div>

      <input
        ref={captureRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(event) => readFile(event.target.files?.[0] ?? null)}
      />
      <input
        ref={uploadRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => readFile(event.target.files?.[0] ?? null)}
      />
    </div>
  );
};
