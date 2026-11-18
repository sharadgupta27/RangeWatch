import { ChevronLeft, ChevronRight, ExternalLink, LocateFixed, X } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'

import type { LayerSet, NativeRangeOut, OccurrenceRow } from '@/api/types'
import { DeckMap } from '@/components/maps/DeckMap'
import { focusLayers, locateZoom, nativeRangeLayer, occurrenceLayer } from '@/components/maps/layers'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { formatDate, formatNumber } from '@/lib/utils'
import type { ViewState } from '@/store/mapStore'

/** Public page of the original record (a plain link; the app never calls these services). */
export function recordUrl(row: Pick<OccurrenceRow, 'source' | 'record_id'>): string | null {
  if (row.source === 'GBIF') return `https://www.gbif.org/occurrence/${row.record_id}`
  if (row.source === 'iNaturalist') return `https://www.inaturalist.org/observations/${row.record_id}`
  return null
}

/** OpenStreetMap at the record, for place names, land use and roads the basemap leaves out. */
export function osmUrl(latitude: number, longitude: number, zoom: number): string {
  const z = Math.round(Math.min(18, Math.max(3, zoom + 1)))
  return `https://www.openstreetmap.org/?mlat=${latitude.toFixed(5)}&mlon=${longitude.toFixed(5)}#map=${z}/${latitude.toFixed(5)}/${longitude.toFixed(5)}`
}

/** Initial camera on a record, framed by its coordinate uncertainty. */
export function locateView(row: Pick<OccurrenceRow, 'latitude' | 'longitude' | 'coordinate_uncertainty_m'>): ViewState {
  return {
    longitude: row.longitude,
    latitude: row.latitude,
    zoom: locateZoom(row.latitude, row.coordinate_uncertainty_m),
    pitch: 0,
    bearing: 0,
  }
}

interface Props {
  record: OccurrenceRow
  layers: LayerSet | undefined
  nativeRange: NativeRangeOut | null | undefined
  /** 1-based position of the record in the current page, and the page size. */
  position: number
  count: number
  onPrev?: () => void
  onNext?: () => void
  onClose: () => void
  /** The record's annotation control (shared with the table row). */
  annotate: ReactNode
}

/**
 * Map panel for reviewing one occurrence: zoomed to the record with its uncertainty circle,
 * neighbouring records and the native-range polygon, plus the annotation control.
 */
export function OccurrenceLocator(props: Props) {
  const { record } = props
  // Keyed by the record so the camera jumps to each newly selected record.
  return <LocatorBody key={`${record.source}:${record.record_id}`} {...props} />
}

function LocatorBody({ record, layers, nativeRange, position, count, onPrev, onNext, onClose, annotate }: Props) {
  const home = useMemo(() => locateView(record), [record])
  const [view, setView] = useState<ViewState>(home)

  const mapLayers = useMemo(
    () => [
      ...(nativeRange
        ? [
            nativeRangeLayer({
              id: 'locator-native-range',
              geometry: nativeRange.geometry as GeoJSON.MultiPolygon,
              status: nativeRange.status,
            }),
          ]
        : []),
      ...(layers ? [occurrenceLayer({ layer: layers.occurrences, maxYear: null })] : []),
      ...focusLayers({
        longitude: record.longitude,
        latitude: record.latitude,
        uncertaintyM: record.coordinate_uncertainty_m,
      }),
    ],
    [layers, nativeRange, record],
  )

  const link = recordUrl(record)
  return (
    <section
      aria-label={`Location of ${record.source} record ${record.record_id}`}
      className="overflow-hidden rounded-xl border bg-card"
    >
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2 border-b px-3 py-2 text-xs">
        <div className="space-y-0.5">
          <div className="font-medium">
            {record.source} <span className="font-mono text-muted-foreground">#{record.record_id}</span>
            <Badge variant="secondary" className="ml-2">
              {record.range_label}
            </Badge>
          </div>
          <div className="font-mono text-muted-foreground">
            {record.latitude.toFixed(5)}, {record.longitude.toFixed(5)} ·{' '}
            {record.coordinate_uncertainty_m == null
              ? 'uncertainty not reported'
              : `± ${formatNumber(record.coordinate_uncertainty_m)} m`}{' '}
            · {formatDate(record.event_date)}
          </div>
          <div className="flex flex-wrap gap-3 pt-0.5">
            {link && (
              <a href={link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary hover:underline">
                Open record on {record.source} <ExternalLink className="size-3" />
              </a>
            )}
            <a
              href={osmUrl(record.latitude, record.longitude, view.zoom)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              Open in OpenStreetMap <ExternalLink className="size-3" />
            </a>
          </div>
        </div>
        <div className="min-w-44">{annotate}</div>
        <div className="ml-auto flex items-center gap-1">
          <span className="mr-1 tabular-nums text-muted-foreground">
            {position} / {count}
          </span>
          <Button variant="outline" size="icon" className="size-7" onClick={onPrev} disabled={!onPrev} aria-label="Previous record">
            <ChevronLeft />
          </Button>
          <Button variant="outline" size="icon" className="size-7" onClick={onNext} disabled={!onNext} aria-label="Next record">
            <ChevronRight />
          </Button>
          <Button variant="outline" size="icon" className="size-7" onClick={() => setView(home)} aria-label="Zoom back to the record" title="Zoom back to the record">
            <LocateFixed />
          </Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={onClose} aria-label="Close map">
            <X />
          </Button>
        </div>
      </div>
      <div className="h-80">
        <DeckMap layers={mapLayers} viewState={view} onViewStateChange={setView} />
      </div>
      <p className="border-t px-3 py-1.5 text-[11px] text-muted-foreground">
        Yellow ring: this record; shaded circle: its reported coordinate uncertainty. Other dots are
        the species' records (individual points from zoom 7); the outline is the native-range polygon.
      </p>
    </section>
  )
}
