"""Call the API and print its answer in a readable form.

    python src/call_api.py                    # a known attack
    python src/call_api.py --from-test 0      # row 0 of KDDTest+
    python src/call_api.py --from-test 3 --raw

This is how another program talks to the service: send JSON, read JSON. The /docs page does
the same thing, it just shows a lot of other boxes around the answer.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

# Same connection as the example on the /docs page: a neptune SYN flood from KDDTest+.
ATTACK_EXAMPLE = {
    "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ",
    "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0, "root_shell": 0,
    "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
    "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 229, "srv_count": 10, "serror_rate": 0.0, "srv_serror_rate": 0.0,
    "rerror_rate": 1.0, "srv_rerror_rate": 1.0, "same_srv_rate": 0.04, "diff_srv_rate": 0.06,
    "srv_diff_host_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 10,
    "dst_host_same_srv_rate": 0.04, "dst_host_diff_srv_rate": 0.06,
    "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0,
}


def post(url: str, payload: dict) -> tuple[int, dict]:
    """Send `payload` as JSON, return the status code and the answer."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:            # 422 and friends carry a JSON body too
        return error.code, json.load(error)
    except urllib.error.URLError as error:
        raise SystemExit(f"cannot reach {url}: {error.reason}\n"
                         "Is the server running? python -m uvicorn api:app --reload")


def row_from_test(index: int) -> tuple[dict, str]:
    """One connection from KDDTest+, with its true label, so the answer can be checked."""
    from data import FEATURES, load
    test = load("KDDTest+.txt")
    if not 0 <= index < len(test):
        raise SystemExit(f"--from-test must be between 0 and {len(test) - 1}")
    row = test.iloc[index]
    values = {k: (v.item() if hasattr(v, "item") else v) for k, v in row[FEATURES].items()}
    return values, str(row["label"])


def show(answer: dict, truth: str | None) -> None:
    verdict = "ALERT " if answer["prediction"] == "attack" else "quiet "
    print(f"\n{verdict} {answer['prediction']:<7} probability {answer['attack_probability']:.4f}"
          f"   threshold {answer['threshold']}")
    if truth is not None:
        correct = (truth != "normal") == (answer["prediction"] == "attack")
        print(f"        truth:  {truth:<14} {'correct' if correct else 'WRONG'}")
    print("\nwhy:")
    for factor in answer.get("top_factors", []):
        arrow = "->attack" if factor["contribution"] > 0 else "->normal"
        print(f"  {factor['feature']:<28} = {str(factor['value']):<10} "
              f"{factor['contribution']:+7.2f}  {arrow}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--from-test", type=int, default=None, metavar="N",
                    help="send row N of KDDTest+ instead of the built-in example")
    ap.add_argument("--top-k", type=int, default=5, help="how many factors to show")
    ap.add_argument("--raw", action="store_true", help="also print the raw JSON answer")
    args = ap.parse_args()

    if args.from_test is None:
        connection, truth = ATTACK_EXAMPLE, None
    else:
        connection, truth = row_from_test(args.from_test)

    status, answer = post(f"{args.url}/explain?top_k={args.top_k}", connection)
    if status != 200:
        print(f"the server refused the request ({status}):")
        for problem in answer.get("detail", []):
            print(f"  field {problem.get('loc', ['?'])[-1]}: {problem.get('msg')}")
        raise SystemExit(1)

    show(answer, truth)
    if args.raw:
        print("\nraw answer:")
        print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
