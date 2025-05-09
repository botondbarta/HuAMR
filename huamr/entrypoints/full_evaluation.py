import os
import pandas as pd
import subprocess
import click

from huamr.utils.amr_validator import AMRValidator

@click.command()
@click.argument('data_path')
@click.argument('ref_column')
@click.argument('pred_column')
@click.option('--frame-arg-descr', default=None, help="Path to the frame argument description file (required if --amr-validation is enabled).")
@click.option('--amr-validation', is_flag=True, default=False, help="Enable AMR validation. If enabled, frame_arg_descr is required.")
@click.option('--evaluation-script', default='./evaluation.sh', help="Path to the evaluation script.")
def main(data_path, ref_column, pred_column, frame_arg_descr, amr_validation, evaluation_script):
    if amr_validation and not frame_arg_descr:
        raise click.UsageError("--frame-arg-descr is required when --amr-validation is enabled.")

    amrvalidator = AMRValidator(frame_arg_descr) if amr_validation else None

    df = pd.read_csv(data_path)

    if amr_validation:
        df = df[df.apply(lambda x: amrvalidator.validate(x[pred_column]), axis=1)]
        df = df[df.apply(lambda x: amrvalidator.validate(x[ref_column]), axis=1)]

    gold_amr_content = '\n\n'.join(df[ref_column])
    parsed_amr_content = '\n\n'.join(df[pred_column])

    gold_amr_path = 'gold_amr.txt'
    parsed_amr_path = 'parsed_amr.txt'

    with open(gold_amr_path, 'w', encoding='utf-8') as gold_file:
        gold_file.write(gold_amr_content)

    with open(parsed_amr_path, 'w', encoding='utf-8') as parsed_file:
        parsed_file.write(parsed_amr_content)

    try:
        result = subprocess.run(
            [evaluation_script, parsed_amr_path, gold_amr_path],
            capture_output=True,
            text=True
        )
        print("Evaluation Output:")
        print(result.stdout)
        print("Evaluation Errors (if any):")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        print(e.output)

    finally:
        if os.path.exists(gold_amr_path):
            os.remove(gold_amr_path)
        if os.path.exists(parsed_amr_path):
            os.remove(parsed_amr_path)

if __name__ == "__main__":
    main()