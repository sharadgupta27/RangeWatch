import { Loader2, Trash2 } from 'lucide-react'
import { useState } from 'react'

import { useDeleteSpecies, usePermissions } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

/** Whether the typed confirmation matches the species name (case and spacing ignored). */
export function confirmsName(typed: string, scientificName: string): boolean {
  const norm = (s: string) => s.trim().replace(/\s+/g, ' ').toLowerCase()
  return norm(typed).length > 0 && norm(typed) === norm(scientificName)
}

/**
 * Remove a species that is not needed, with all its records, models and maps. Curators
 * only; the user types the name to confirm. The server refuses while a job is running.
 */
export function DeleteSpeciesButton({
  taxonKey,
  scientificName,
  compact = false,
  disabled = false,
  onDeleted,
}: Readonly<{
  taxonKey: number
  scientificName: string
  compact?: boolean
  disabled?: boolean
  onDeleted?: () => void
}>) {
  const { curate } = usePermissions()
  const remove = useDeleteSpecies()
  const [open, setOpen] = useState(false)
  const [typed, setTyped] = useState('')
  if (!curate) return null
  const ok = confirmsName(typed, scientificName)
  return (
    <Popover
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) {
          setTyped('')
          remove.reset()
        }
      }}
    >
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size={compact ? 'icon' : 'sm'}
          className={compact ? 'size-7 text-muted-foreground hover:text-destructive' : 'text-destructive'}
          disabled={disabled}
          aria-label={`Delete ${scientificName}`}
          title={disabled ? 'Discard the running job first' : 'Delete this species and all its data'}
        >
          <Trash2 />
          {!compact && 'Delete species'}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-3 text-xs">
        <p className="font-medium text-destructive">Delete {scientificName}?</p>
        <p className="text-muted-foreground">
          Removes its occurrence records, native range, all model versions with their maps and
          bulletins, reference data, annotations and alerts. This cannot be undone. MLflow run
          history is kept.
        </p>
        <div className="space-y-1">
          <Label htmlFor={`confirm-${taxonKey}`}>Type the species name to confirm</Label>
          <Input
            id={`confirm-${taxonKey}`}
            value={typed}
            placeholder={scientificName}
            onChange={(e) => setTyped(e.target.value)}
            autoComplete="off"
          />
        </div>
        {remove.isError && <p className="text-destructive">{remove.error.message}</p>}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={!ok || remove.isPending}
            onClick={() =>
              remove.mutate(taxonKey, {
                onSuccess: () => {
                  setOpen(false)
                  onDeleted?.()
                },
              })
            }
          >
            {remove.isPending && <Loader2 className="animate-spin" />} Delete permanently
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
