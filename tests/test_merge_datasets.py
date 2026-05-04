#!/usr/bin/env python3
"""Unit tests for merge_datasets.py script."""
import json
import os
import tempfile
import pytest
from pathlib import Path
from scripts.merge_datasets import (
    load_jsonl,
    save_jsonl,
    deduplicate,
    validate_sample,
    DATASET_GROUPS,
)


class TestLoadSaveJsonl:
    """Test JSONL loading and saving functions."""
    
    def test_load_jsonl_basic(self, tmp_path):
        """Test basic JSONL loading."""
        test_file = tmp_path / "test.jsonl"
        data = [
            {"instruction": "Test 1", "input": "Input 1", "output": "Output 1"},
            {"instruction": "Test 2", "input": "Input 2", "output": "Output 2"},
        ]
        with open(test_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        loaded = load_jsonl(test_file)
        assert len(loaded) == 2
        assert loaded[0]["instruction"] == "Test 1"
        assert loaded[1]["instruction"] == "Test 2"
    
    def test_load_jsonl_nonexistent(self, tmp_path):
        """Test loading non-existent file returns empty list."""
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []
    
    def test_load_jsonl_with_empty_lines(self, tmp_path):
        """Test loading handles empty lines."""
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write('{"instruction": "Test 1"}\n')
            f.write("\n")  # empty line
            f.write('{"instruction": "Test 2"}\n')
        
        loaded = load_jsonl(test_file)
        assert len(loaded) == 2
    
    def test_save_jsonl_basic(self, tmp_path):
        """Test basic JSONL saving."""
        test_file = tmp_path / "output.jsonl"
        data = [
            {"instruction": "Test 1", "input": "Input 1", "output": "Output 1"},
        ]
        
        count = save_jsonl(data, test_file)
        assert count == 1
        assert test_file.exists()
        
        loaded = load_jsonl(test_file)
        assert len(loaded) == 1
        assert loaded[0]["instruction"] == "Test 1"


class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_deduplicate_basic(self):
        """Test basic deduplication."""
        data = [
            {"instruction": "Test", "input": "Input A", "output": "Output 1"},
            {"instruction": "Test", "input": "Input A", "output": "Output 2"},  # duplicate
            {"instruction": "Test", "input": "Input B", "output": "Output 3"},  # unique
        ]
        result = deduplicate(data)
        assert len(result) == 2
    
    def test_deduplicate_all_same(self):
        """Test deduplication when all items are same."""
        data = [
            {"instruction": "Test", "input": "Input", "output": "Output"}
            for _ in range(5)
        ]
        result = deduplicate(data)
        assert len(result) == 1
    
    def test_deduplicate_all_unique(self):
        """Test deduplication when all items are unique."""
        data = [
            {"instruction": f"Test {i}", "input": f"Input {i}", "output": f"Output {i}"}
            for i in range(5)
        ]
        result = deduplicate(data)
        assert len(result) == 5
    
    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        result = deduplicate([])
        assert result == []


class TestValidation:
    """Test sample validation."""
    
    def test_validate_sample_with_instruction(self):
        """Test validation passes with instruction."""
        sample = {"instruction": "Test", "input": "", "output": ""}
        assert validate_sample(sample) is True
    
    def test_validate_sample_with_prompt(self):
        """Test validation passes with prompt."""
        sample = {"prompt": "Test prompt", "completion": "Test"}
        assert validate_sample(sample) is True
    
    def test_validate_sample_empty(self):
        """Test validation fails with empty instruction and prompt."""
        sample = {"input": "Test", "output": ""}
        assert validate_sample(sample) is False
    
    def test_validate_sample_mixed_case(self):
        """Test validation with mixed valid/invalid fields."""
        valid = {"instruction": "Test", "output": ""}
        invalid = {"input": "Only input", "output": "Only output"}
        assert validate_sample(valid) is True
        assert validate_sample(invalid) is False


class TestDatasetGroups:
    """Test dataset group configuration."""
    
    def test_dataset_groups_exist(self):
        """Test all dataset groups are defined."""
        expected_groups = ["core", "synthetic", "github", "extra", "codesearchnet"]
        for group in expected_groups:
            assert group in DATASET_GROUPS
    
    def test_dataset_groups_have_keys(self):
        """Test each group has dataset keys."""
        for group_name, datasets in DATASET_GROUPS.items():
            assert len(datasets) > 0, f"Group {group_name} is empty"
            for name, path in datasets.items():
                assert name, f"Empty key in group {group_name}"
                assert path, f"Empty path for {name}"
    
    def test_core_group(self):
        """Test core dataset group has expected datasets."""
        core = DATASET_GROUPS["core"]
        expected = ["apps_train", "codealpaca", "humaneval", "mbpp"]
        for key in expected:
            assert key in core
    
    def test_synthetic_group(self):
        """Test synthetic dataset group has expected datasets."""
        synthetic = DATASET_GROUPS["synthetic"]
        expected = ["synthetic", "synthetic2", "synthetic3", "synthetic4"]
        for key in expected:
            assert key in synthetic
    
    def test_github_group(self):
        """Test GitHub dataset group has expected datasets."""
        github = DATASET_GROUPS["github"]
        expected = ["github_code", "more_repos", "flask_samples", "gunicorn_samples"]
        for key in expected:
            assert key in github


class TestIntegration:
    """Integration tests for merge functionality."""
    
    def test_merge_two_datasets(self, tmp_path):
        """Test merging two small datasets."""
        # Create test datasets
        ds1 = tmp_path / "ds1.jsonl"
        ds2 = tmp_path / "ds2.jsonl"
        
        with open(ds1, "w") as f:
            f.write(json.dumps({"instruction": "Test 1", "input": "A", "output": "1"}) + "\n")
        
        with open(ds2, "w") as f:
            f.write(json.dumps({"instruction": "Test 2", "input": "B", "output": "2"}) + "\n")
        
        # Load and combine (simulating merge)
        data1 = load_jsonl(ds1)
        data2 = load_jsonl(ds2)
        combined = data1 + data2
        
        assert len(combined) == 2
    
    def test_full_pipeline_with_dedup(self, tmp_path):
        """Test full pipeline with loading, dedup, saving."""
        # Create test data with duplicates
        data = [
            {"instruction": "Test", "input": "A", "output": "1"},
            {"instruction": "Test", "input": "A", "output": "1"},  # duplicate
            {"instruction": "Test", "input": "B", "output": "2"},
        ]
        
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        
        save_jsonl(data, input_file)
        loaded = load_jsonl(input_file)
        
        assert len(loaded) == 3
        
        deduped = deduplicate(loaded)
        assert len(deduped) == 2
        
        save_jsonl(deduped, output_file)
        reloaded = load_jsonl(output_file)
        
        assert len(reloaded) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])