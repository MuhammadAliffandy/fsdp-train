#!/usr/bin/env python3
"""Train one matched full-cohort loss arm using the validated research trainer."""
from fsdp_loss_cohort import configure, adapt_model, finish_summary, extend_parser
from train_fsdp_research import main


if __name__ == '__main__':
    main(configure, adapt_model, finish_summary, extend_parser)
