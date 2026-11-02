/**
 * Named aliases over the generated OpenAPI component schemas.
 * These are *derived* from `schema.d.ts` (never hand-written shapes) — regenerate with
 * `npm run gen:api` whenever the backend contract changes.
 */
import type { components } from './schema'

type S = components['schemas']

export type TaxonSearchResult = S['TaxonSearchResult']
export type SpeciesSummary = S['SpeciesSummary']
export type SpeciesDetail = S['SpeciesDetail']
export type SpeciesStatus = S['SpeciesStatus']
export type ModelSummary = S['ModelSummary']
export type ModelVersionOut = S['ModelVersionOut']
export type ModelVersionDetail = S['ModelVersionDetail']
export type SeverityResult = S['SeverityResult']
export type SeverityConfigModel = S['SeverityConfigModel']
export type NativeRangeOut = S['NativeRangeOut']
export type NativeRangeUpdate = S['NativeRangeUpdate']
export type JobOut = S['JobOut']
export type OccurrenceRow = S['OccurrenceRow']
export type OccurrencePage = S['OccurrencePage']
export type Timeline = S['Timeline']
export type LayerSet = S['LayerSet']
export type RasterLayer = S['RasterLayer']
export type VectorLayer = S['VectorLayer']
export type LegendEntry = S['LegendEntry']
export type BulletinStatus = S['BulletinStatus']
export type BioclimInfo = S['BioclimInfo']
export type ReproducibilityInfo = S['ReproducibilityInfo']
export type GeoJSONMultiPolygon = S['GeoJSONMultiPolygon']
export type GeoJSONPolygon = S['GeoJSONPolygon']
export type ValidationReport = S['ValidationReport']
export type ValidationSpecies = S['ValidationSpecies']
export type ValidationRegion = S['ValidationRegion']
export type HiresLayer = S['HiresLayer']
export type ClimateCrossCheckOut = S['ClimateCrossCheckOut']
export type BackgroundInfo = S['BackgroundInfo']
export type ExtrapolationSummaryOut = S['ExtrapolationSummaryOut']
export type ExtrapolationDiagnosticOut = S['ExtrapolationDiagnosticOut']
export type DataQualityOut = S['DataQualityOut']
export type DataQualityTestOut = S['DataQualityTestOut']
export type SamplingEffortOut = S['SamplingEffortOut']
export type ReferenceDatasetOut = S['ReferenceDatasetOut']
export type ReferenceEvaluationOut = S['ReferenceEvaluationOut']
export type IndependentMetricsOut = S['IndependentMetricsOut']
export type DatasetUploadResult = S['DatasetUploadResult']
/** Multipart fields of POST /species/{taxon_key}/datasets (the file is sent separately). */
export type DatasetUploadFields = Omit<S['Body_uploadReferenceDataset'], 'file'>
