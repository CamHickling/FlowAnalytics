import os
import subprocess
import argparse
import sys


def find_gopro_pair(session_dir):
    gopro_dir = os.path.join(session_dir, 'gopro_footage')
    if not os.path.isdir(gopro_dir):
        return None, None

    mp4s = [f for f in os.listdir(gopro_dir) if f.lower().endswith('.mp4')]
    if len(mp4s) < 2:
        return None, None

    front = None
    side = None
    for f in mp4s:
        lf = f.lower()
        if 'front' in lf and front is None:
            front = f
        if 'side' in lf and side is None:
            side = f

    if front and side:
        return os.path.join(gopro_dir, front), os.path.join(gopro_dir, side)

    return os.path.join(gopro_dir, mp4s[0]), os.path.join(gopro_dir, mp4s[1])


def main(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Root does not exist: {root_dir}")
        return

    if not sys.stdin.isatty():
        print('This script requires an interactive terminal.')
        return

    sessions = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    sessions.sort()

    py = sys.executable or 'python'

    for session in sessions:
        left, right = find_gopro_pair(session)
        print('\nSession:', os.path.basename(session))
        if not left or not right:
            print('  No GoPro pair found (skipping)')
            continue
        print(f'  Left: {left}')
        print(f'  Right: {right}')

        while True:
            ans = input('  Launch visual sync for this session? (y/n/q): ').strip().lower()
            if ans in ('y', 'n', 'q'):
                break
        if ans == 'q':
            print('Aborting batch.')
            return
        if ans == 'n':
            continue

        cmd = [py, os.path.join(os.path.dirname(__file__), 'visual_sync.py'), '--left', left, '--right', right, '--session', session]
        # run visual_sync as a blocking subprocess so the user can interact with its window
        subprocess.run(cmd)

    print('\nAll sessions processed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch visual_sync.py for each session interactively')
    parser.add_argument(
        '--root',
        default=os.environ.get(
            'FLOW_ANALYTICS_DATA_ROOT',
            r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data",
        ),
        help='Root folder with session subfolders',
    )
    args = parser.parse_args()
    main(args.root)
