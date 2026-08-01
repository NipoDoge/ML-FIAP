import pytest
from pydantic import ValidationError

from src.schemas.roles_schemas import role, role_update
from src.schemas.users_schemas import users_create, users_update, usersGetData


class TestUsersSchemas:
    def test_user_create_requires_password(self):
        payload = {
            "name": "Gabriel",
            "email": "gabriel@example.com",
        }

        with pytest.raises(ValidationError):
            users_create.model_validate(payload)

    def test_user_create_accepts_valid_email(self):
        payload = {
            "name": "Gabriel",
            "email": "gabriel@example.com",
            "password": "secret123",
        }

        user = users_create.model_validate(payload)

        assert user.name == "Gabriel"
        assert user.email == "gabriel@example.com"
        assert user.password == "secret123"
        assert user.active is True

    def test_user_model_rejects_invalid_email(self):
        payload = {
            "name": "Gabriel",
            "email": "not-an-email",
            "password": "secret123",
        }

        with pytest.raises(ValidationError):
            users_create.model_validate(payload)

    def test_user_update_accepts_partial_payload(self):
        payload = {
            "name": "Gabriel Silva",
            "email": "gabriel.silva@example.com",
        }

        user_update = users_update.model_validate(payload)

        assert user_update.name == "Gabriel Silva"
        assert user_update.email == "gabriel.silva@example.com"

    def test_users_get_data_defaults_active_true(self):
        payload = {
            "name": "Gabriel",
            "email": "gabriel@example.com",
        }

        user_data = usersGetData.model_validate(payload)

        assert user_data.active is True
        assert user_data.role is None


class TestRoleSchemas:
    def test_role_requires_description_and_active(self):
        with pytest.raises(ValidationError):
            role.model_validate({"description": "Admin"})

        with pytest.raises(ValidationError):
            role.model_validate({"active": True})

    def test_role_model_accepts_complete_payload(self):
        payload = {
            "description": "Admin",
            "active": True,
        }

        result = role.model_validate(payload)

        assert result.description == "Admin"
        assert result.active is True

    def test_role_update_accepts_partial_payload(self):
        payload = {
            "description": "Operator",
        }

        result = role_update.model_validate(payload)

        assert result.description == "Operator"
        assert result.active is None
