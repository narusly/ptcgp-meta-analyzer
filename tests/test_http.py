from ptcgp.http import HttpClient, parse_ratelimit


def test_parse_ratelimit():
    assert parse_ratelimit('"50-in-5min"; r=49; t=300') == (49, 300)
    assert parse_ratelimit(None) is None
    assert parse_ratelimit("garbage") is None


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_client_waits_for_the_next_window_when_no_request_is_left():
    fake = FakeClock()
    client = HttpClient(log=lambda _: None, sleep=fake.sleep, clock=fake.clock)
    client._note_limits({"RateLimit": '"50-in-5min"; r=0; t=10'})
    client._wait_turn()
    assert fake.sleeps == [11]


def test_client_does_not_wait_while_requests_remain():
    fake = FakeClock()
    client = HttpClient(log=lambda _: None, sleep=fake.sleep, clock=fake.clock)
    client._note_limits({"RateLimit": '"50-in-5min"; r=12; t=200'})
    client._wait_turn()
    assert fake.sleeps == []
