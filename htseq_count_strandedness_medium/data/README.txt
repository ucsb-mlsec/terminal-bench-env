Subject: RNA-seq Dataset - Strand Specificity Unknown

Hey there,

Thanks for helping me out with this analysis! I received this RNA-seq dataset from a collaborator last week, but unfortunately they forgot to document which library preparation protocol they used. I know, I know - it happens more often than it should in busy labs.

The issue is that we need to determine the strand-specificity setting before we can proceed with proper downstream analysis. As you probably know, modern RNA-seq library prep kits can produce unstranded libraries (where reads map equally to both strands), forward stranded libraries (like the dUTP method or NSR protocols where read 1 maps to the transcript strand), or reverse stranded libraries (where read 1 maps to the opposite strand of the transcript).

I've pulled out a representative subset of aligned reads and saved them in the SAM file for you to examine. The GTF file contains the gene annotations for the reference genome, which you'll need to compare against the read alignments to figure out the strandedness pattern.

This determination is actually critical - if we get the strand-specificity wrong, our gene expression quantification will be completely off, especially for overlapping genes or antisense transcripts. We might end up counting reads from the wrong strand and drawing incorrect biological conclusions.

Let me know what you find! The data should have a clear pattern once you look at how the reads align relative to the annotated features.

Cheers