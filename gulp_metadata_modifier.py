import argparse
import glob
import logging
import re
import shutil
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Pattern
from typing import Union

import numpy as np
import pandas as pd
import ujson

__all__ = [
    "main",
    "modify_metadata",
    "modify_all_gulp_dirs",
    "GulpExampleId",
    "GulpMetaDataDict",
    "GulpFrameInfo",
    "GulpExampleMetaDict",
    "GulpDirectoryMetaDict",
]

LOG = logging.getLogger(__name__)

GulpExampleId = str
GulpMetaDataDict = Dict[GulpExampleId, Any]
GulpFrameInfo = List[List[int]]
GulpExampleMetaDict = Dict[str, Union[GulpFrameInfo, List[GulpMetaDataDict]]]
GulpDirectoryMetaDict = Dict[GulpExampleId, GulpExampleMetaDict]


parser = argparse.ArgumentParser(
    description="Update the metadata in a gulp directory from a given dataframe",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "annotations_pkl",
    type=Path,
    help="Path to pickled DataFrame containing all annotations. The index must "
    "correspond to the ID of the gulped examples. All the columns' values will be "
    "copied into the first item of the metadata field for that example.",
)
parser.add_argument(
    "gulp_dirs",
    type=Path,
    nargs="+",
    metavar="GULP_DIR",
    help="Gulp directories to update",
)
parser.add_argument(
    "--drop-unknown",
    action="store_true",
    help="Drop entries for examples not found in annotations DataFrame",
)
parser.add_argument(
    "--ignore-missing-examples",
    action="store_true",
    help="If an example is missing from annotations_pkl then this script will raise "
    "an error unless --drop-unknown is set, however if you simply want to leave "
    "that metadata untouched when --drop-unkonwn is not set, then set this flag.",
)
parser.add_argument(
    "--disable-backup",
    action="store_false",
    dest="enable_backup",
    help="Disable creation of gmeta_X.bak files when updating metadata",
)


def main(args=None) -> None:
    if args is None:
        args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    annotations = pd.read_pickle(args.annotations_pkl)

    def update_annotation(
        example_id: GulpExampleId, old_example_metadata: GulpMetaDataDict
    ) -> Optional[GulpMetaDataDict]:
        try:
            example_metadata = annotations.loc[example_id].to_dict()
        except KeyError as e:
            if args.drop_unknown or args.ignore_missing_examples:
                return None
            raise e

        # Update the existing metadata in case have extra items that we would lose if
        # we simply replaced it.
        updated_metadata = dict(old_example_metadata)
        updated_metadata.update(example_metadata)
        return updated_metadata

    for gulp_dir in args.gulp_dirs:
        modify_metadata(
            gulp_dir,
            transform_func=update_annotation,
            drop_nones=args.drop_unknown,
            backup=args.enable_backup,
        )


def modify_all_gulp_dirs(
    gulp_dir_root: Path,
    transform_func: Callable[[GulpExampleId, GulpExampleMetaDict], GulpExampleMetaDict],
    gulp_dir_pattern: Pattern = re.compile(".*gulp.*"),
) -> None:
    """
    Apply ``transform_func`` to all gulp metadata within all gulp directories
    matching ``gulp_dir_pattern`` in ``gulp_dir_root``.

    Args:
        gulp_dir_root: Root directory to search for gulp directories
        transform_func: User provided function to transform an example's metadata
        gulp_dir_pattern: A directory below ``gulp_dir_root`` is considered a gulp
            directory if it matches this pattern.

    See Also:
        :py:func:`modify_metadata`
    """
    gulp_dirs = [
        child_dir
        for child_dir in gulp_dir_root.iterdir()
        if gulp_dir_pattern.search(child_dir.name)
    ]
    for gulp_dir in gulp_dirs:
        modify_metadata(gulp_dir, transform_func)


def modify_metadata(
    gulp_dir: Union[Path, str],
    transform_func: Callable[
        [GulpExampleId, GulpMetaDataDict], Optional[GulpMetaDataDict]
    ],
    *,
    drop_nones: bool = False,
    backup: bool = True
) -> None:
    """Update the metadata in ``gulp_dir`` according to the user provided function
    ``transform_func`` which takes in a single example's id and metadata and
    transforms it.

    Args:
        gulp_dir: Gulp directory containing ``.gmeta`` and ``.gulp`` files.
        transform_func: User provided function to transform the metadata of an example
            in some way. This should take in the example id and old metadata and return
            either ``None`` if the segment is to be dropped (if ``drop_nones=True`` or
            to be left unchanged otherwise) or the updated metadata dict.
        drop_nones: If set and ``transform_func`` returns None, then remove the segment
            from the gulp meta dict.
        backup: Make ``.bak`` files for all ``.gmeta`` files.
    """
    if isinstance(gulp_dir, str):
        gulp_dir = Path(gulp_dir)
    meta_dicts: Dict[Path, GulpDirectoryMetaDict] = dict()
    for meta_path in _find_meta_files(gulp_dir):
        with meta_path.open(mode="r", encoding="utf-8") as f:
            meta_dicts[meta_path] = ujson.load(f)

    LOG.info("Modifying {}".format(gulp_dir))
    ignored = []
    for meta_path, meta_dict in meta_dicts.items():
        example_ids = set(meta_dict.keys())
        for example_id in example_ids:
            example_ignored = _update_metadata(
                example_id, meta_dict, transform_func, drop_nones=drop_nones
            )
            if example_ignored:
                ignored.append(example_id)

        if backup:
            _backup_meta_data(meta_path)

        with meta_path.open(mode="w", encoding="utf-8") as f:
            ujson.dump(meta_dict, f)
    _report_action_on_ignored_examples(ignored, drop_nones)


def _report_action_on_ignored_examples(
    ignored: List[GulpExampleId], drop_nones: bool
) -> None:
    if len(ignored) > 0:
        if drop_nones:
            LOG.info("{} examples dropped".format(len(ignored)))
            action = "dropped"
        else:
            LOG.info("{} examples not updated".format(len(ignored)))
            action = "not updated"
        for example_id in ignored:
            LOG.info("{!r} {}".format(example_id, action))


def _backup_meta_data(meta_path: Path) -> None:
    """Backup metadata file with ``.bak`` suffix. If it already exists, then use the
    ``.bakX`` suffix where ``X`` is a number starting from 0 and increasing until
    ``.bakX`` no longer exists.

    Args:
        meta_path: Path to metadata file to backup
    """
    meta_path = meta_path.resolve()
    backup_meta_path = meta_path.parent / (meta_path.name + ".bak")
    i = 0
    while backup_meta_path.exists():
        backup_meta_path = backup_meta_path.with_suffix(".bak{}".format(i))
        i += 1
    shutil.copy(str(meta_path), str(backup_meta_path))


def _update_metadata(
    example_id: GulpExampleId,
    meta_dict: GulpMetaDataDict,
    transform_func,
    *,
    drop_nones=False
) -> bool:
    """
    Args:
        example_id: ID of example to update in ``meta_dict``
        meta_dict:
        transform_func:
        drop_nones:

    Returns:
        False if the example was not handled by transformed_func
    """
    example_metadata = meta_dict[example_id]["meta_data"][0]
    # We have to convert numpy values to python values as the json module can't dump
    # them
    transformed_metadata = transform_func(example_id, example_metadata)
    if transformed_metadata is None and drop_nones:
        del meta_dict[example_id]
    else:
        meta_dict[example_id]["meta_data"] = [_numpy_to_builtin(transformed_metadata)]
    return transformed_metadata is None


def _numpy_to_builtin(obj):
    """
    Convert possibly numpy values into python builtin values. Traverses structures applying the type
    transformations

    Args:
        obj: an arbitrary python object, if it contains numpy values it will be converted into a
        corresponding python value.

    Examples:
        >>> type(_numpy_to_builtin(np.int64(2)))
        <class 'int'>
        >>> type(_numpy_to_builtin(np.int32(2)))
        <class 'int'>
        >>> type(_numpy_to_builtin(np.int16(2)))
        <class 'int'>
        >>> type(_numpy_to_builtin(np.int8(2)))
        <class 'int'>
        >>> type(_numpy_to_builtin(np.int(2)))
        <class 'int'>
        >>> type(_numpy_to_builtin(np.float(2.2)))
        <class 'float'>
        >>> l = [np.int8(2), [np.int64(3)]]
        >>> l_converted = _numpy_to_builtin(l)
        >>> len(l) == len(l_converted)
        True
        >>> type(l_converted[0])
        <class 'int'>
        >>> type(l_converted[1][0])
        <class 'int'>
        >>> d = {'a': np.int8(2), 'b': { 'c': np.int32(2) }}
        >>> d_converted = _numpy_to_builtin(d)
        >>> type(d_converted['a'])
        <class 'int'>
        >>> type(d_converted['b']['c'])
        <class 'int'>
        >>> type(_numpy_to_builtin(np.array([1, 2], dtype=np.int32))[0])
        <class 'int'>
        >>> type(_numpy_to_builtin(np.array([1.2, 2.2], dtype=np.float32))[0])
        <class 'float'>
    """
    if isinstance(obj, np.generic) and np.isscalar(obj):
        return np.asscalar(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        for key in obj.keys():
            value = obj[key]
            obj[key] = _numpy_to_builtin(value)
        return obj
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_builtin(o) for o in obj]
    else:
        return obj


def _find_meta_files(path: Path) -> List[Path]:
    return sorted(list(map(Path, glob.glob(str(path.joinpath("meta*.gmeta"))))))


if __name__ == "__main__":
    main()
