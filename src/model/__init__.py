from src.model.random_forest_classifier import make_model

from src.model.is_name import make_model as is_name_model


def choose_model(task):
    match task:
        case "is_comic_video":
            return make_model()
        case "is_name":
            return is_name_model()
        case "find_comic_name":
            return None
        case _:
            raise ValueError("Wrong task")