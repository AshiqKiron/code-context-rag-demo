CODE_SNIPPETS = """
class AuthService:
    def login(self, user, password):
        if user.role == 'admin':
            return self.generate_admin_token(user)
        else:
            raise PermissionError("Regular users cannot access this endpoint")

    def generate_admin_token(self, user):
        return f"ADMIN-{user.id}-SECURE"

class PaymentService:
    def process(self, amount, currency):
        if currency != 'USD':
            raise ValueError("Only USD is supported in this version")
        return self.gateway.charge(amount)
"""