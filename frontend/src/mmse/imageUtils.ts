/**
 * Client-side image validation + compression for the Q11 Copying photo.
 *
 * The examiner's photo is validated and (if needed) downscaled/re-compressed in
 * the browser BEFORE it is sent to the vision endpoint:
 *
 *  - Accepted MIME types: image/jpeg, image/png, image/webp.
 *  - Max upload size: 10 MB.
 *  - Minimum resolution: 800x600 (low-res photos are REJECTED).
 *  - Max long side: 2048 (larger photos are downscaled, aspect preserved).
 *  - No upscaling ever.
 *  - Encoding: JPEG, quality 0.80–0.85.
 *
 * Only the optimized data URL is sent to `/mmse/copying/evaluate`. Original and
 * optimized dimensions + sizes are reported back (`photoInfo`) so the UI can show
 * what happened; these are never persisted.
 */

export const PHOTO_MAX_BYTES = 10 * 1024 * 1024;
export const PHOTO_MIN_WIDTH = 800;
export const PHOTO_MIN_HEIGHT = 600;
export const PHOTO_MAX_LONG_SIDE = 2048;
export const PHOTO_JPEG_QUALITY = 0.82;

export const PHOTO_LOW_RESOLUTION_MESSAGE =
  'Image resolution is too low for reliable assessment. Please take a clearer photo.';

export const PHOTO_ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export interface PhotoInfo {
  original: { width: number; height: number; bytes: number };
  optimized: { width: number; height: number; bytes: number };
  /** True when the image was downscaled/re-encoded before sending. */
  wasOptimized: boolean;
}

export interface ProcessedPhoto {
  dataUrl: string;
  info: PhotoInfo;
}

/** Rough data-URL byte estimate: base64 length * 3/4. */
function dataUrlBytes(dataUrl: string): number {
  const comma = dataUrl.indexOf(',');
  const b64 = comma === -1 ? dataUrl : dataUrl.slice(comma + 1);
  return Math.round((b64.length * 3) / 4);
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image could not be decoded.'));
    img.src = dataUrl;
  });
}

export function isAcceptedMime(file: File): boolean {
  if (PHOTO_ACCEPTED_TYPES.includes(file.type)) return true;
  return /\.(jpe?g|png|webp)$/i.test(file.name);
}

/**
 * Validate + optimize a Q11 photo. Resolves to a processed data URL (always
 * JPEG, possibly downscaled) plus dimension/size info, or throws an Error with a
 * user-facing message (classify as an `upload` error kind).
 */
export async function processPhoto(file: File): Promise<ProcessedPhoto> {
  if (!isAcceptedMime(file)) {
    throw new Error('Unsupported image type. Use JPEG, PNG, or WebP.');
  }
  if (file.size > PHOTO_MAX_BYTES) {
    throw new Error('The image exceeds the maximum allowed size (10 MB).');
  }

  const rawDataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
    reader.onerror = () => reject(new Error('The photo could not be read.'));
    reader.readAsDataURL(file);
  });

  const img = await loadImage(rawDataUrl);

  const originalWidth = img.naturalWidth;
  const originalHeight = img.naturalHeight;
  if (originalWidth < PHOTO_MIN_WIDTH || originalHeight < PHOTO_MIN_HEIGHT) {
    throw new Error(PHOTO_LOW_RESOLUTION_MESSAGE);
  }

  // Scale down only when the long side exceeds the cap; never upscale.
  const longSide = Math.max(originalWidth, originalHeight);
  const scale = longSide > PHOTO_MAX_LONG_SIDE ? PHOTO_MAX_LONG_SIDE / longSide : 1;
  const targetWidth = Math.round(originalWidth * scale);
  const targetHeight = Math.round(originalHeight * scale);
  const wasOptimized = scale < 1 || file.type !== 'image/jpeg';

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is not supported in this browser.');
  ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

  const optimized = canvas.toDataURL('image/jpeg', PHOTO_JPEG_QUALITY);
  const info: PhotoInfo = {
    original: { width: originalWidth, height: originalHeight, bytes: file.size },
    optimized: {
      width: targetWidth,
      height: targetHeight,
      bytes: dataUrlBytes(optimized),
    },
    wasOptimized,
  };
  return { dataUrl: optimized, info };
}
