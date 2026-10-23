"""Spawn workers preserve serial conversion without downloaded/prepared LAFAN."""

import argparse
import multiprocessing

import numpy as np


def test_spawned_preparation_matches_serial_conversion(tmp_path):
    from supermimic.data.motion_convert import convert_csv_to_npz
    from supermimic.data.prepare import guarded_convert, init

    jobs = []
    for i in range(2):
        source = tmp_path / f"motion_{i}.csv"
        raw = np.zeros((16, 36))
        raw[:, 0] = np.arange(16) / 30 * (i + 1)
        raw[:, 2] = 0.8
        raw[:, 6] = 1  # xyzw identity quaternion
        raw[:, 7] = np.linspace(0, 0.1 * (i + 1), 16)
        np.savetxt(source, raw, delimiter=",")
        convert_csv_to_npz(source, tmp_path / f"serial_{i}.npz", 30, 50)
        jobs.append((str(source), str(tmp_path / f"spawn_{i}.npz")))

    options = argparse.Namespace(robot="g1", input_fps=30.0, skip_invalid=False)
    with multiprocessing.get_context("spawn").Pool(
        2, initializer=init, initargs=(options,)
    ) as pool:
        reports = pool.map(guarded_convert, jobs)
    for i, report in enumerate(reports):
        assert report["clip"] == f"motion_{i}"
        assert report["frames"] == 26
        with (
            np.load(tmp_path / f"serial_{i}.npz", allow_pickle=False) as expected,
            np.load(tmp_path / f"spawn_{i}.npz", allow_pickle=False) as actual,
        ):
            assert set(actual.files) == set(expected.files)
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
            assert actual["joint_pos"].shape == (26, 29)
            np.testing.assert_allclose(actual["body_lin_vel_w"][:, 0, 0], i + 1)
