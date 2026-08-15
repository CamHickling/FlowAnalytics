import os
import cv2
import argparse
import subprocess
from datetime import datetime
import imageio_ffmpeg
import numpy as np


def get_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return ret, frame


def render_final(g1, g2, offset, output_file):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    filter_complex = "[0:v]scale=960:-1[v0];[1:v]scale=960:-1[v1];[v0][v1]hstack=inputs=2[vout]"
    if offset > 0:
        command = [
            ffmpeg_exe, '-y',
            '-i', g1,
            '-ss', str(offset), '-i', g2,
            '-filter_complex', filter_complex,
            '-map', '[vout]', '-map', '0:a',
            '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-shortest',
            output_file
        ]
    else:
        command = [
            ffmpeg_exe, '-y',
            '-ss', str(abs(offset)), '-i', g1,
            '-i', g2,
            '-filter_complex', filter_complex,
            '-map', '[vout]', '-map', '0:a',
            '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-shortest',
            output_file
        ]
    subprocess.run(command)


def visual_sync(left_video, right_video, session_path=None):
    capA = cv2.VideoCapture(left_video)
    capB = cv2.VideoCapture(right_video)
    if not capA.isOpened() or not capB.isOpened():
        print('Failed to open one or both videos')
        return

    fpsA = capA.get(cv2.CAP_PROP_FPS) or 30.0
    fpsB = capB.get(cv2.CAP_PROP_FPS) or 30.0
    framesA = int(capA.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    framesB = int(capB.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    posA = 0
    posB = 0
    anchorA = None
    anchorB = None

    help_text = [
        "Controls:",
        "h/l: move both -/+1 frame",
        "H/L: move both -/+10 frames",
        "j/k: move left -/+1 frame",
        "n/m: move right -/+1 frame",
        "1: set anchor for left video",
        "2: set anchor for right video",
        "r: render final synced video (asks confirm)",
        "q: quit"
    ]

    window_name = 'Visual Sync - press q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        retA, frameA = get_frame(capA, max(0, min(posA, framesA - 1)))
        retB, frameB = get_frame(capB, max(0, min(posB, framesB - 1)))

        if not retA or not retB:
            # show blank if a frame can't be read
            if not retA:
                frameA = np.zeros((480, 640, 3), dtype=np.uint8)
            if not retB:
                frameB = np.zeros((480, 640, 3), dtype=np.uint8)

        # resize to same height
        h = 480
        if frameA is not None:
            ah, aw = frameA.shape[:2]
            scaleA = h / ah
            frameA = cv2.resize(frameA, (int(aw * scaleA), h))
        if frameB is not None:
            bh, bw = frameB.shape[:2]
            scaleB = h / bh
            frameB = cv2.resize(frameB, (int(bw * scaleB), h))

        combined = np.hstack([frameA, frameB])

        # overlay status text
        status = f"A: {posA}/{framesA} ({posA/fpsA:.2f}s)  B: {posB}/{framesB} ({posB/fpsB:.2f}s)"
        if anchorA is not None:
            status += f"  AnchorA: {anchorA}"
        if anchorB is not None:
            status += f"  AnchorB: {anchorB}"
        cv2.putText(combined, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y = 40
        for line in help_text:
            cv2.putText(combined, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            posA = max(0, posA - 1); posB = max(0, posB - 1)
        elif key == ord('l'):
            posA = min(framesA - 1, posA + 1); posB = min(framesB - 1, posB + 1)
        elif key == ord('H'):
            posA = max(0, posA - 10); posB = max(0, posB - 10)
        elif key == ord('L'):
            posA = min(framesA - 1, posA + 10); posB = min(framesB - 1, posB + 10)
        elif key == ord('j'):
            posA = max(0, posA - 1)
        elif key == ord('k'):
            posA = min(framesA - 1, posA + 1)
        elif key == ord('n'):
            posB = max(0, posB - 1)
        elif key == ord('m'):
            posB = min(framesB - 1, posB + 1)
        elif key == ord('1'):
            anchorA = posA
            print(f"Anchor A set to frame {anchorA} ({anchorA/fpsA:.3f}s)")
        elif key == ord('2'):
            anchorB = posB
            print(f"Anchor B set to frame {anchorB} ({anchorB/fpsB:.3f}s)")
        elif key == ord('r'):
            # require both anchors
            if anchorA is None or anchorB is None:
                print('Please set both anchors (press 1 and 2) before rendering.')
                continue
            timeA = anchorA / fpsA
            timeB = anchorB / fpsB
            offset = timeB - timeA
            print(f"Computed offset (B - A) = {offset:.4f} seconds")
            # save selection to sync_results.txt in session folder if provided
            if session_path:
                result_file = os.path.join(session_path, 'sync_results.txt')
                with open(result_file, 'a', encoding='utf-8') as fh:
                    fh.write(f"{datetime.now().isoformat()}\toption=visual\toffset={offset:.4f}\tvideo={os.path.basename(session_path)}_synced.mp4\tnote=visual_sync\n")
                print(f"Saved selection to {result_file}")

            # ask for confirm render
            resp = input('Render final synced video now using this offset? (y/n): ').strip().lower()
            if resp == 'y':
                outname = os.path.join(session_path or os.getcwd(), f"{os.path.basename(session_path) if session_path else 'session'}_synced.mp4")
                print(f"Rendering final video to: {outname} (this may take some time)")
                render_final(left_video, right_video, offset, outname)
                print('Rendering complete.')
        else:
            # ignore unknown keys
            pass

    capA.release()
    capB.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual sync two videos using OpenCV')
    parser.add_argument('--left', required=True, help='Left video path')
    parser.add_argument('--right', required=True, help='Right video path')
    parser.add_argument('--session', required=False, help='Session folder to save selection (optional)')
    args = parser.parse_args()

    # ensure interactive
    if not os.isatty(0):
        print('This visual tool requires an interactive terminal and a display.')
        raise SystemExit(1)

    visual_sync(args.left, args.right, session_path=args.session)
