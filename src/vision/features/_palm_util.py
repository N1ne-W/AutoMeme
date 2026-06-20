import numpy as np

def is_palm_open(hand_landmarks) -> bool:
    if hand_landmarks is None:
        return False
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    wrist = hand_landmarks.landmark[0]
    opened = 0
    for tid, mid in zip(tip_ids, mcp_ids):
        tip = hand_landmarks.landmark[tid]
        mcp = hand_landmarks.landmark[mid]
        d_tip = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        d_mcp = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
        if d_tip > d_mcp:
            opened += 1
    return opened >= 4
