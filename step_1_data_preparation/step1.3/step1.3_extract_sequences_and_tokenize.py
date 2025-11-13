#!/usr/bin/env python3
"""
Step 1.3: Extract DNA Sequences and Create 3-mer Tokenization for DNABERT

This script performs two tasks:
1. Extracts DNA sequences from reference genome for panel regions
2. Converts sequences to 3-mer tokens for DNABERT input

Author: PDAC cfDNA Deconvolution Project
Date: 2025-11-13
"""

import sys
import gzip
import argparse
from pathlib import Path
from collections import defaultdict


def load_fasta_index(fasta_path):
    """
    Load chromosome sequences from FASTA file.
    Handles both gzipped and uncompressed FASTA files.
    
    Args:
        fasta_path: Path to reference genome FASTA file
        
    Returns:
        Dictionary mapping chromosome names to sequences
    """
    print(f"Loading reference genome from: {fasta_path}")
    sequences = {}
    current_chrom = None
    current_seq = []
    
    # Determine if file is gzipped
    open_func = gzip.open if str(fasta_path).endswith('.gz') else open
    mode = 'rt' if str(fasta_path).endswith('.gz') else 'r'
    
    with open_func(fasta_path, mode) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous chromosome
                if current_chrom and current_seq:
                    sequences[current_chrom] = ''.join(current_seq)
                    print(f"  Loaded {current_chrom}: {len(sequences[current_chrom]):,} bp")
                
                # Start new chromosome
                current_chrom = line[1:].split()[0]  # Take first part of header
                current_seq = []
            else:
                current_seq.append(line.upper())
            
            # Progress indicator
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} lines...")
        
        # Save last chromosome
        if current_chrom and current_seq:
            sequences[current_chrom] = ''.join(current_seq)
            print(f"  Loaded {current_chrom}: {len(sequences[current_chrom]):,} bp")
    
    print(f"Total chromosomes loaded: {len(sequences)}")
    return sequences


def extract_sequence(sequences, chrom, start, end, strand='+'):
    """
    Extract DNA sequence for a genomic region.
    
    Args:
        sequences: Dictionary of chromosome sequences
        chrom: Chromosome name (e.g., 'chr1')
        start: Start position (0-based)
        end: End position (0-based, exclusive)
        strand: '+' or '-'
        
    Returns:
        DNA sequence string (uppercase)
    """
    if chrom not in sequences:
        return None
    
    # Ensure coordinates are within bounds
    chrom_len = len(sequences[chrom])
    start = max(0, start)
    end = min(end, chrom_len)
    
    if start >= end:
        return None
    
    seq = sequences[chrom][start:end]
    
    # Reverse complement if on negative strand
    if strand == '-':
        complement = str.maketrans('ATCG', 'TAGC')
        seq = seq.translate(complement)[::-1]
    
    return seq


def dna_to_3mer(sequence):
    """
    Convert DNA sequence to 3-mer tokens for DNABERT.
    
    Example:
        ATCGATCG -> ATC TCG CGA GAT ATC TCG
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Space-separated 3-mer string
    """
    if len(sequence) < 3:
        return ""
    
    kmers = []
    k = 3
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        # Only include kmers with valid nucleotides
        if all(base in 'ATCG' for base in kmer):
            kmers.append(kmer)
    
    return ' '.join(kmers)


def process_bed_file(bed_path, sequences, output_fasta, output_3mer):
    """
    Process BED file: extract sequences and create 3-mer tokens.
    
    Args:
        bed_path: Path to BED file with panel regions
        sequences: Dictionary of chromosome sequences
        output_fasta: Path to output FASTA file
        output_3mer: Path to output 3-mer file
    """
    print(f"\nProcessing BED file: {bed_path}")
    
    regions_processed = 0
    regions_skipped = 0
    total_bases = 0
    
    with open(bed_path, 'r') as bed, \
         open(output_fasta, 'w') as fasta_out, \
         open(output_3mer, 'w') as kmer_out:
        
        # Write header for 3-mer file
        kmer_out.write("region_id\tchrom\tstart\tend\tstrand\tsequence_length\tn_3mers\t3mer_sequence\n")
        
        for line_num, line in enumerate(bed, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 3:
                continue
            
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            
            # Get strand if available (default to +)
            strand = fields[5] if len(fields) > 5 else '+'
            
            # Create region ID
            region_id = f"{chrom}_{start}_{end}"
            
            # Extract sequence
            seq = extract_sequence(sequences, chrom, start, end, strand)
            
            if seq is None or len(seq) == 0:
                regions_skipped += 1
                continue
            
            # Convert to 3-mers
            kmers = dna_to_3mer(seq)
            n_kmers = len(kmers.split()) if kmers else 0
            
            if n_kmers == 0:
                regions_skipped += 1
                continue
            
            # Write to FASTA
            fasta_out.write(f">{region_id}\n")
            fasta_out.write(f"{seq}\n")
            
            # Write to 3-mer file
            kmer_out.write(f"{region_id}\t{chrom}\t{start}\t{end}\t{strand}\t{len(seq)}\t{n_kmers}\t{kmers}\n")
            
            regions_processed += 1
            total_bases += len(seq)
            
            # Progress indicator
            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} regions ({regions_processed:,} successful, {regions_skipped:,} skipped)")
    
    print(f"\nSummary:")
    print(f"  Total regions processed: {regions_processed:,}")
    print(f"  Total regions skipped: {regions_skipped:,}")
    print(f"  Total bases extracted: {total_bases:,}")
    print(f"  Average region length: {total_bases/regions_processed:.1f} bp")


def main():
    parser = argparse.ArgumentParser(
        description='Extract DNA sequences from BED file and create 3-mer tokenization for DNABERT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step1.3_extract_sequences_and_tokenize.py \\
    --bed /path/to/TWIST_blocks.bed \\
    --fasta /path/to/hg38.fa.gz \\
    --output-dir ./output
        """
    )
    
    parser.add_argument(
        '--bed',
        required=True,
        help='Path to BED file with panel regions'
    )
    
    parser.add_argument(
        '--fasta',
        required=True,
        help='Path to reference genome FASTA file (can be gzipped)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./step1.3_output',
        help='Output directory for results (default: ./step1.3_output)'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='panel_sequences',
        help='Prefix for output files (default: panel_sequences)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    bed_path = Path(args.bed)
    fasta_path = Path(args.fasta)
    
    if not bed_path.exists():
        print(f"Error: BED file not found: {bed_path}", file=sys.stderr)
        sys.exit(1)
    
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output files
    output_fasta = output_dir / f"{args.output_prefix}.fa"
    output_3mer = output_dir / f"{args.output_prefix}_3mers.txt"
    
    print("="*80)
    print("Step 1.3: Extract DNA Sequences and Create 3-mer Tokenization")
    print("="*80)
    print(f"Input BED file: {bed_path}")
    print(f"Reference genome: {fasta_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output FASTA: {output_fasta}")
    print(f"Output 3-mers: {output_3mer}")
    print("="*80)
    
    # Load reference genome
    sequences = load_fasta_index(fasta_path)
    
    # Process BED file
    process_bed_file(bed_path, sequences, output_fasta, output_3mer)
    
    print("\n" + "="*80)
    print("Step 1.3 Complete!")
    print("="*80)
    print(f"FASTA sequences: {output_fasta}")
    print(f"3-mer tokens: {output_3mer}")
    print("\nNext step: Use these files for Step 1.4 (Create Training Dataset)")


if __name__ == "__main__":
    main()
