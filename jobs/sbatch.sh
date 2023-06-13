sbatch --export=ALL,motion=$1 -o out/deep_mimic_$1.out -e out/deep_mimic_$1.err -J deep_mimic_$1 jobs/deep_mimic.sh
