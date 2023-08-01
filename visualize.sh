experiment="jaw6g0q7"
target_label="ERBB2_mRNA"
classes="mRNA_High WT"
# target_label="ERBB2_Protein"
# classes="Protein_High WT"

# target_label="Grade"
# classes="1 2 3"

orig_pwd=$(pwd)

for cls in $classes; do
    cd ../wanshi-utils && python -m wanshi.visualizations.roc \
        --outpath=../roc_$(basename $experiment)_$cls.svg \
        --target-label=$target_label \
        --true-class=$cls \
        /mnt/bulk/gwoelflein/georg-transformers/output/$experiment/fold*/patient-preds.csv
    cd $orig_pwd
done