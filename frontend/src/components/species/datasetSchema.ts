import { z } from 'zod'

const MAX_MB = 25
const ACCEPTED = ['.csv', '.tsv', '.txt', '.geojson', '.json', '.gpkg', '.zip'] as const
export const ACCEPTED_UPLOAD_TYPES = ACCEPTED.join(',')

const optionalText = z.string().trim().max(200)

/** Client-side checks mirroring the backend's upload limits (the server re-validates). */
export const datasetUploadSchema = z.object({
  file: z
    .custom<File | null>((v) => v === null || v instanceof File)
    .refine((f) => f !== null, 'Choose a file')
    .refine((f) => !f || f.size <= MAX_MB * 1024 * 1024, `Files are limited to ${MAX_MB} MB`)
    .refine(
      (f) => !f || ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext)),
      `Use one of ${ACCEPTED.join(', ')}`,
    ),
  name: z.string().trim().min(1, 'Name the dataset').max(200),
  citation: optionalText,
  license: optionalText,
  note: z.string().trim().max(2000),
  longitude_column: optionalText,
  latitude_column: optionalText,
  species_column: optionalText,
  date_column: optionalText,
  uncertainty_column: optionalText,
  status_column: optionalText,
  record_id_column: optionalText,
})

export type DatasetUploadForm = z.input<typeof datasetUploadSchema>

export const EMPTY_UPLOAD: DatasetUploadForm = {
  file: null,
  name: '',
  citation: '',
  license: '',
  note: '',
  longitude_column: '',
  latitude_column: '',
  species_column: '',
  date_column: '',
  uncertainty_column: '',
  status_column: '',
  record_id_column: '',
}

export const COLUMN_FIELDS = [
  ['longitude_column', 'Longitude'],
  ['latitude_column', 'Latitude'],
  ['species_column', 'Species name'],
  ['date_column', 'Date'],
  ['uncertainty_column', 'Uncertainty (m)'],
  ['status_column', 'Presence / absence'],
  ['record_id_column', 'Record id'],
] as const
