#!/usr/bin/env bash
# Interactive submitter for the batch jobs in jobs/.
#
# Lists every jobs/*.sh along with the resources its #SBATCH header asks for and
# the command it ends up running, then hands the ones you pick to sbatch.
#
#   ./batch.sh                  pick from the menu
#   ./batch.sh train_funet      submit by name, no prompt
#   ./batch.sh -n all           show the sbatch commands without submitting
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
JOBS_DIR="$ROOT/jobs"
LOG_DIR="$ROOT/logs"

DRY_RUN=0

if [ -t 1 ]; then B=$(printf '\033[1m'); D=$(printf '\033[2m'); R=$(printf '\033[0m')
else B=""; D=""; R=""; fi

usage() {
  cat <<EOF
Usage: batch.sh [-n|--dry-run] [job ...]

Submits the job scripts in jobs/ with sbatch. With no job named, lists what is
available and prompts. Names are the jobs/*.sh filenames, .sh optional.

  -n, --dry-run   Print the sbatch commands instead of running them.
  -h, --help      This message.
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; break ;;
    -*)           printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    *)            break ;;
  esac
done

names=()
paths=()
for f in "$JOBS_DIR"/*.sh; do
  [ -f "$f" ] || continue          # no match leaves the glob unexpanded
  names[${#names[@]}]=$(basename "$f" .sh)
  paths[${#paths[@]}]=$f
done

if [ ${#names[@]} -eq 0 ]; then
  printf 'No job scripts found in %s\n' "$JOBS_DIR" >&2
  exit 1
fi

# Pull one option out of a script's #SBATCH header. Slurm accepts both "-p foo"
# and "--partition=foo" and these scripts mix the two, so look for either. Pass
# "-" as the short name for options that only have a long form.
sbatch_opt() {
  awk -v short="-$2" -v long="--$3=" '
    $1 == "#SBATCH" {
      for (i = 2; i <= NF; i++) {
        if (short != "-" && $i == short && i < NF) { print $(i + 1); exit }
        if (index($i, long) == 1) { print substr($i, length(long) + 1); exit }
      }
    }
  ' "$1"
}

# Everything above the poetry lines is identical boilerplate in every script
# (module load, setup.sh), so the poetry lines are the only interesting part.
job_command() {
  awk '/^[[:space:]]*poetry run /{
         sub(/^[[:space:]]+/, "")
         printf "%s%s", sep, $0
         sep = "; "
       }
       END { print "" }' "$1"
}

list_jobs() {
  local i part cpus gpus mem tl cmd
  printf '\n  Available Jobs:\n\n'
  for ((i = 0; i < ${#names[@]}; i++)); do
    part=$(sbatch_opt "${paths[$i]}" p partition)
    cpus=$(sbatch_opt "${paths[$i]}" c cpus-per-task)
    gpus=$(sbatch_opt "${paths[$i]}" G gpus)
    mem=$(sbatch_opt  "${paths[$i]}" - mem)
    tl=$(sbatch_opt   "${paths[$i]}" t time)
    cmd=$(job_command "${paths[$i]}")
    printf '  %s%2d)%s %s%s%s\n' "$D" "$((i + 1))" "$R" "$B" "${names[$i]}" "$R"
    printf '      %s%s · %s cpu · %s gpu · %s · %s%s\n' \
      "$D" "${part:-default partition}" "${cpus:-1}" "${gpus:-0}" \
      "${mem:-default mem}" "${tl:-partition default time}" "$R"
    printf '      %s\n\n' "${cmd:-(no poetry command found)}"
  done
}

index_of() {
  local want=${1%.sh} i
  for ((i = 0; i < ${#names[@]}; i++)); do
    if [ "${names[$i]}" = "$want" ]; then printf '%s' "$i"; return 0; fi
  done
  return 1
}

# Turns "1 3", "all" or "train_funet" into indices on stdout, one per line,
# in menu order and deduplicated. Non-zero exit means a token was bad.
parse_selection() {
  local tok idx seen=" " out=""
  for tok in $1; do
    case $tok in
      all|a)     for ((idx = 0; idx < ${#names[@]}; idx++)); do out="$out$idx "; done ;;
      ''|*[!0-9]*)
        if idx=$(index_of "$tok"); then out="$out$idx "
        else printf '  no such job: %s\n' "$tok" >&2; return 1; fi ;;
      *)
        if [ "$tok" -ge 1 ] && [ "$tok" -le ${#names[@]} ]; then out="$out$((tok - 1)) "
        else printf '  out of range: %s\n' "$tok" >&2; return 1; fi ;;
    esac
  done
  for ((idx = 0; idx < ${#names[@]}; idx++)); do
    case $out in *" $idx "*|"$idx "*)
      case $seen in *" $idx "*) ;; *) printf '%s\n' "$idx"; seen="$seen$idx " ;; esac ;;
    esac
  done
}

submit() {
  local name=${names[$1]} script=${paths[$1]} out id
  # Both flags below deliberately override the #SBATCH directives in the script.
  # Slurm gives the command line precedence, and both directives are wrong as
  # written:
  #   --chdir     the scripts say ~/dev/fhr-analysis, but sbatch parses #SBATCH
  #               lines itself with no shell involved, so the ~ never expands.
  #               This points at wherever the repo actually is instead.
  #   --job-name  the default is the filename *with* .sh, which would make the
  #               %x in the -o pattern produce logs/train_funet.sh_123.out.
  set -- sbatch --chdir="$ROOT" --job-name="$name" "$script"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  %s\n' "$*"
    return 0
  fi
  if ! out=$("$@" 2>&1); then
    printf '  %-18s %sFAILED%s\n' "$name" "$B" "$R" >&2
    printf '    %s\n' "$out" >&2
    return 1
  fi
  id=${out##* }                     # "Submitted batch job 12345"
  printf '  %-18s job %-10s %slogs/%s_%s.out%s\n' "$name" "$id" "$D" "$name" "$id" "$R"
}

if [ $# -gt 0 ]; then
  interactive=0
  selection="$*"
else
  interactive=1
  list_jobs
  while :; do
    printf '  Submit which? %s(numbers, names, "all", "q" to quit)%s ' "$D" "$R"
    if ! read -r reply; then printf '\n'; exit 0; fi
    case $(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]') in
      q|quit|exit) exit 0 ;;
      '')          continue ;;
    esac
    selection=${reply//,/ }
    parse_selection "$selection" >/dev/null && break
  done
fi

chosen=$(parse_selection "${selection//,/ }") || exit 2

if [ "$interactive" -eq 1 ]; then
  printf '\n  Submitting:\n'
  for i in $chosen; do printf '    %s\n' "${names[$i]}"; done
  printf '  Proceed? [y/N] '
  read -r ok || ok=""
  case $ok in
    y|Y|yes|YES|Yes) ;;
    *) printf '  Nothing submitted.\n'; exit 0 ;;
  esac
fi

if [ "$DRY_RUN" -eq 0 ]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    printf '\nsbatch not found -- run this on a cluster login node.\n' >&2
    printf 'Use --dry-run to see what would be submitted.\n' >&2
    exit 1
  fi
  # The -o/-e patterns in every job script write into logs/, and Slurm will not
  # create a missing output directory: the job just dies with nowhere to say so.
  mkdir -p "$LOG_DIR"
fi

printf '\n'
rc=0
for i in $chosen; do
  submit "$i" || rc=1
done
exit $rc
