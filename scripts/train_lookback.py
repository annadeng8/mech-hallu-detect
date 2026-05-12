import joblib
from src.models.lookback import LookbackLens


def main():
    lens = LookbackLens()
    joblib.dump(lens, "outputs/lookback.pkl")
    print("Saved Lookback model")


if __name__ == "__main__":
    main()