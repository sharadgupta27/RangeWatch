import {
  DrawPolygonMode,
  EditableGeoJsonLayer,
  ModifyMode,
  TranslateMode,
  ViewMode,
} from '@deck.gl-community/editable-layers'
import { useSelector } from '@tanstack/react-store'
import type { Feature, FeatureCollection, MultiPolygon, Polygon, Position } from 'geojson'
import { CheckCircle2, Eraser, Move, PenLine, PlusSquare, RotateCcw, Save } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import type { NativeRangeOut, VectorLayer } from '@/api/types'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { formatDate } from '@/lib/utils'
import { mapActions, mapStore } from '@/store/mapStore'

import { DeckMap } from './DeckMap'
import { COLORS, occurrenceLayer } from './layers'

type EditMode = 'modify' | 'translate' | 'draw' | 'view'
const MODES = {
  modify: ModifyMode,
  translate: TranslateMode,
  draw: DrawPolygonMode,
  view: ViewMode,
} as const

/** Split a (Multi)Polygon into one editable Polygon feature per part. */
export function toFeatureCollection(geom: Polygon | MultiPolygon): FeatureCollection<Polygon> {
  const parts: Position[][][] = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates
  return {
    type: 'FeatureCollection',
    features: parts.map((coordinates) => ({
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates },
    })),
  }
}

/** Merge all edited polygon features back into a single MultiPolygon for the API. */
export function toMultiPolygon(fc: FeatureCollection): MultiPolygon | null {
  const coords: Position[][][] = []
  for (const f of fc.features) {
    if (f.geometry?.type === 'Polygon') coords.push(f.geometry.coordinates)
    if (f.geometry?.type === 'MultiPolygon') coords.push(...f.geometry.coordinates)
  }
  return coords.length ? { type: 'MultiPolygon', coordinates: coords } : null
}

interface Props {
  nativeRange: NativeRangeOut
  occurrences: VectorLayer
  saving: boolean
  onSave: (geometry: MultiPolygon, confirm: boolean) => void
}

export function NativeRangeEditor({ nativeRange, occurrences, saving, onSave }: Props) {
  const viewState = useSelector(mapStore, (s) => s.viewState)
  const serverFc = useMemo(
    () => toFeatureCollection(nativeRange.geometry as MultiPolygon),
    [nativeRange.geometry],
  )
  const [fc, setFc] = useState<FeatureCollection>(serverFc)
  const [mode, setMode] = useState<EditMode>('modify')
  const [selected, setSelected] = useState<number[]>([0])
  useEffect(() => setFc(serverFc), [serverFc])

  const dirty = JSON.stringify(fc) !== JSON.stringify(serverFc)
  const geometry = toMultiPolygon(fc)

  const editLayer = useMemo(
    () =>
      new EditableGeoJsonLayer({
        id: 'native-range-editor',
        data: fc as FeatureCollection,
        mode: MODES[mode],
        selectedFeatureIndexes: mode === 'draw' ? [] : selected,
        onEdit: ({ updatedData, editType }) => {
          setFc(updatedData as FeatureCollection)
          if (editType === 'addFeature') {
            setSelected([(updatedData as FeatureCollection).features.length - 1])
            setMode('modify')
          }
        },
        onClick: (info: { index: number }) => {
          if (mode !== 'draw' && info.index >= 0) setSelected([info.index])
        },
        pickable: true,
        getFillColor: COLORS.nativeRangeFill,
        getLineColor: COLORS.nativeRangeLine,
        getLineWidth: 2,
        lineWidthUnits: 'pixels',
        getEditHandlePointColor: [250, 250, 250, 255],
        getEditHandlePointRadius: 4,
        editHandlePointRadiusUnits: 'pixels',
      }),
    [fc, mode, selected],
  )

  const layers = useMemo(
    () => [occurrenceLayer({ layer: occurrences, maxYear: null }), editLayer],
    [occurrences, editLayer],
  )

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
      <div className="relative h-[560px] overflow-hidden rounded-xl border">
        <DeckMap
          layers={layers}
          viewState={viewState}
          onViewStateChange={mapActions.setViewState}
          doubleClickZoom={false}
          getCursor={({ isDragging }) => (mode === 'draw' ? 'crosshair' : isDragging ? 'grabbing' : 'grab')}
        >
          <div className="absolute top-3 left-3">
            <ToggleGroup type="single" value={mode} onValueChange={(v) => v && setMode(v as EditMode)}>
              <ToggleGroupItem value="modify" aria-label="Edit vertices">
                <PenLine /> Edit vertices
              </ToggleGroupItem>
              <ToggleGroupItem value="translate" aria-label="Move polygon">
                <Move /> Move
              </ToggleGroupItem>
              <ToggleGroupItem value="draw" aria-label="Draw additional area">
                <PlusSquare /> Add area
              </ToggleGroupItem>
            </ToggleGroup>
          </div>
          {mode === 'draw' && (
            <div className="absolute top-14 left-3 rounded-md border bg-card/90 px-3 py-1.5 text-xs shadow backdrop-blur">
              Click to add vertices, double-click to close the polygon.
            </div>
          )}
        </DeckMap>
      </div>

      <div className="flex flex-col gap-4">
        <Alert variant={nativeRange.status === 'confirmed' ? 'info' : 'warning'}>
          <CheckCircle2 />
          <AlertTitle>
            {nativeRange.status === 'confirmed' ? 'Confirmed native range' : 'Review required'}
          </AlertTitle>
          <AlertDescription>
            The native range decides which records count as native vs. introduced, and therefore
            whether Model B (native + invaded) or Model A (native only) is trained. It is never
            finalised automatically — review it against GISD / POWO / literature and edit as needed
            before confirming.
          </AlertDescription>
        </Alert>

        <div className="space-y-2 rounded-xl border p-4 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Status</span>
            <Badge variant={nativeRange.status === 'confirmed' ? 'success' : 'warning'}>
              {nativeRange.status}
            </Badge>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Source</span>
            <span className="truncate font-mono text-[11px]">{nativeRange.source}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Last updated</span>
            <span>{formatDate(nativeRange.updated_ts, true)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Polygon parts</span>
            <span>{fc.features.length}</span>
          </div>
          {nativeRange.note && (
            <p className="border-t pt-2 leading-relaxed text-muted-foreground">{nativeRange.note}</p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm" disabled={!dirty} onClick={() => setFc(serverFc)}>
            <RotateCcw /> Revert
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={fc.features.length < 2 || selected.length === 0}
            onClick={() => {
              setFc({
                ...fc,
                features: fc.features.filter((_: Feature, i: number) => !selected.includes(i)),
              })
              setSelected([0])
            }}
          >
            <Eraser /> Remove part
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={!geometry || saving}
            onClick={() => geometry && onSave(geometry, false)}
          >
            <Save /> Save draft
          </Button>
          <Button size="sm" disabled={!geometry || saving} onClick={() => geometry && onSave(geometry, true)}>
            <CheckCircle2 /> Confirm &amp; train
          </Button>
        </div>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Confirming queues a pipeline run. If a model already exists, editing a confirmed range
          flags it for retraining.
        </p>
      </div>
    </div>
  )
}
